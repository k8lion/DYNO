import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict
from robustbench.data import load_imagenetc
import math
import h5py
import os
import shutil
import copy
import random

CORRUPTIONS = ("pixelate", "contrast", "jpeg_compression", "elastic_transform")

def corr_greedy_prune(activations: torch.Tensor = None, num2prune: int = 0, crosscorr: bool = False, prenorm: bool = True, device="cpu", limit_ratio=-1, zeroprune=True):
    if activations is None:
        return None
    to_prune = []
    activations = activations.to(device)
    if len(activations.shape) > 2:
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
    if limit_ratio > 0 and activations.shape[0]/activations.shape[1] > limit_ratio:
        sampleindices = torch.randperm(activations.shape[0])[:activations.shape[1]*limit_ratio]
        activations = activations[sampleindices]
    if prenorm:
        if zeroprune:
            activations[:,torch.norm(activations,dim=0)==0.0] = 1.0
        activations = torch.nan_to_num(activations/torch.norm(activations,dim=0),0)
    indices = list(range(activations.shape[1]))
    for i in range(len(to_prune)):
        activations = torch.cat((activations[:,:indices.index(to_prune[i])],activations[:,indices.index(to_prune[i])+1:]),dim=1)
        indices.remove(to_prune[i])
    total = len(indices)
    if crosscorr:
        corr = torch.nan_to_num(torch.corrcoef(activations.t()),0)
    else:
        corr = activations.t() @ activations
    while(num2prune > 0):
        neuron = torch.argmax(torch.norm(corr, dim=1)).item()
        if neuron in to_prune:
            num2prune = 0
            break
        to_prune.append(neuron)
        corr[:,neuron] = 0
        corr[neuron,:] = 0
        num2prune -= 1
        total -= 1
        activations = torch.cat((activations[:,:indices.index(neuron)],activations[:,indices.index(neuron)+1:]),dim=1)
        indices.remove(neuron)
    return to_prune

def zero_prune(activations: torch.Tensor = None):
    return (torch.norm(activations, dim=0) == 0).nonzero().flatten().tolist()


def get_lr_weights(model, loader, criterion, device="cpu", flip_labels=False, eb_criterion=False):
    # Only considering "weight" and "bias" gradients 
    layer_names = [
        n for n, _ in model.named_parameters() if "bn" not in n
    ]  # and "bias" not in n]

    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    xent_grads, entropy_grads = [], []
    counter = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if flip_labels:
            y = 9-y  # Reverse labels; quick way to simulate last-layer setting
        logits = model(x)

        loss_xent = criterion(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters()#, retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])
        counter += 1
        if counter >= 3:
            break

    def get_grad_norms(model, grads):
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            if eb_criterion:
                tmp = (grad*grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
                _metrics[name] = tmp.mean().item()
            else:
                _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics

def perform_surgery(optimizer, model, loader, criterion, layer_weights, device="cpu", flip_labels=False, eb_criterion=False, reinit=True):
    weights = get_lr_weights(model, loader, criterion, device, flip_labels, eb_criterion) #get average grad norm/weight norm for each tensor over 5 batches
    max_weight = max(weights.values())
    for k, v in weights.items(): 
        weights[k] = v / max_weight
    layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
    print("Weights: ", weights)
    #tune_metrics['layer_weights'] = layer_weights
    params = defaultdict()
    for n, p in model.named_parameters():
        if "bn" not in n:
            params[n] = p 
    params_weights = []
    if reinit:
        for param, weight in weights.items():
            params_weights.append({"params": params[param], "lr": weight*optimizer.defaults["lr"]})
        optimizer = torch.optim.Adam(params_weights, lr=optimizer.defaults["lr"], weight_decay=optimizer.defaults["weight_decay"])
    else:
        for paramgroup in optimizer.param_groups:
            for param, weight in weights.items():
                if param in paramgroup['params']:
                    paramgroup['lr'] = weight*optimizer.defaults["lr"]
    return optimizer

def train(model, train_loader, optimizer, criterion, epochs=-1, iterations=-1, val_loader=None,
          verbose=False, val_verbose=True, device="cpu", regression=False, val_acts = False, surgical = False):
    model.train()
    val_losses = []
    val_accs = []
    if val_acts:
        acts = defaultdict(list)
    else:
        acts = None
    if iterations == -1:
        iterations = epochs * len(train_loader)
    iteration = 0
    samples = 0
    print(f"Training for {iterations} iterations")
    while iteration < iterations:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # if regression:
            #     output = output.flatten()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            samples += len(target)
            if (batch_idx+1) % 50 == 0 and verbose:
                print('Train iter: {} ({} samples) \tLoss: {:.6f}'.format(
                    iteration, samples, loss.item()))
            iteration += 1
            if iteration >= iterations:
                break
    if val_loader is not None:
        if val_verbose: 
            print("Validation: ", end = "")
        loss, acc = test(model, val_loader, criterion, device=device, regression=regression, verbose=val_verbose)
        val_losses.append(loss)
        val_accs.append(acc)
        if val_acts:
            for key, value in model.activations.items():
                acts[key].append(value.cpu())
    return val_losses, val_accs, acts

def test(model, test_loader, criterion, epochs=1.0, device="cpu", regression=False, verbose=True, metrics=False, act_only=False, layer_index=-1):
    model.eval()
    if metrics and hasattr(model, "activations"):
        for key in model.activations.keys():
            model.activations[key] = torch.Tensor().to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterations = epochs * len(test_loader)
        print(f"Testing for {iterations} iterations")
        iteration = 0
        while iteration < iterations:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, layer_index=layer_index)
                # if regression:
                #     output = output.flatten()
                if not act_only:
                    test_loss += criterion(output, target).item() 
                    if not regression:
                        if len(output.shape) == 2:
                            pred = output.argmax(dim=1, keepdim=True) 
                        else:
                            pred = output > 0
                        correct += pred.eq(target.view_as(pred)).sum().item()
                total += output.shape[0]
                iteration += 1
                if iteration >= iterations:
                    break

    if not act_only:
        test_loss /= total
        
        if verbose:
            print('Average loss: {:.4f}'.format(test_loss), end="")
            if not regression:
                print(', Accuracy: {}/{} ({:.2f}%)'.format(correct, total,
                    100. * correct / total), end="")
            if not metrics:
                print()
    
    return test_loss, correct/total if not act_only else None


def mnist_dataset():
    dataset = datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    return dataset

"""
Turn a dataset into train, val, and test split dataloaders
"""
def split_dataset(X, y=None, X_test=None, y_test=None, val_size=0.1, test_size=0.1, batch_size=128, shuffle_val=False):
    if y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        dataset = X
    if X_test is None:
        dataset, test_set = torch.utils.data.random_split(dataset, lengths=[int((1-test_size)*len(dataset)), int(test_size*len(dataset))])
    elif y_test is not None:
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
    else:
        test_set = X_test
    train_set, val_set = torch.utils.data.random_split(dataset, lengths=[int((1-(val_size/(1-test_size)))*len(dataset)), int((val_size/(1-test_size))*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_val)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


class Galaxy10Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dir='~', transform=transforms.ToTensor(), mode="train"):
        if mode == "train":
            file = str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"
        else:
            file = str(path_to_dir)+"/data/Galaxy10_DECals_test.h5"
        
        f = h5py.File(file, 'r')
        labels, images = f['labels'], f['images']

        self.x = images
        self.y = torch.from_numpy(labels[:]).long()
        self.num_samples = len(images)

        self.transform = transform

    def __getitem__(self, item):
        img = self.x[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[item]

    def __len__(self):
        return self.num_samples

def get_galaxy10_dataloaders(path_to_dir = "~", validation_split=0.1, batch_size=-1, dim=224, random_seed=42):
    if batch_size < 0:
        batch_size = 32 
    if not os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"):
        if os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals.h5"):
            print("making dataset")
            make_galaxy10_traintest(path_to_dir)
        else:
            print("No data found")
            return None, None, None
    totensor = transforms.ToTensor()
    transform = transforms.Compose([
            totensor,
            transforms.Resize(dim),
            transforms.Normalize([0.16683201, 0.16196689, 0.15829432], [0.12819551, 0.11757845, 0.11118137]),
        ])
    galaxy10_train = Galaxy10Dataset(mode='train', transform=transform, path_to_dir=path_to_dir)
    dataset_size = galaxy10_train.num_samples
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(galaxy10_train, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(galaxy10_train, batch_size=batch_size,
                                                    sampler=valid_sampler)

    galaxy10_test = Galaxy10Dataset(mode='test', transform=transform, path_to_dir=path_to_dir)
    test_loader = torch.utils.data.DataLoader(galaxy10_test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def make_galaxy10_traintest(path_to_dir = "..", test_split=0.1, seed=42):
    np.random.seed(seed)
    file = str(path_to_dir)+"/data/Galaxy10_DECals.h5"
    f = h5py.File(file, 'r')
    print(f.keys())
    labels, images = f['ans'], f['images']
    inds = np.arange(len(labels))
    np.random.shuffle(inds)
    split_ind = int(np.floor(test_split * len(inds)))
    tv_inds, test_inds = sorted(inds[split_ind:]), sorted(inds[:split_ind])
    tv_f = h5py.File(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5", 'w')
    tv_f.create_dataset('images', data=images[tv_inds,:,:,:])
    tv_f.create_dataset('labels', data=labels[tv_inds])
    tv_f.close()
    test_f = h5py.File(str(path_to_dir)+"/data/Galaxy10_DECals_test.h5", 'w')
    test_f.create_dataset('images', data=images[test_inds,:,:,:])
    test_f.create_dataset('labels', data=labels[test_inds])
    test_f.close()

def get_imagenet_dataloader(path_to_dir = "~", batch_size=-1, dim=224):
    #wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    if batch_size < 0:
        batch_size = 32 
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256 if dim == 224 else dim+10),
            transforms.CenterCrop(dim),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    if not os.path.exists(str(path_to_dir)+"/data/imagenet/val"):
        print("No data found")
        return None
    valset = datasets.ImageFolder(str(path_to_dir)+"/data/imagenet/val", transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
    return val_loader

def get_continual_cifar100_dataloaders(path_to_dir="~", batch_size=-1, val_size = 0.1, num_tasks=10, seed=42):
    if batch_size < 0:
        batch_size = 32 
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
    if not os.path.exists(str(path_to_dir)+"/data/cifar100"):
        print("No data found.")
        return None, None
    tv_dataset = datasets.CIFAR100(root=str(path_to_dir)+"/data/cifar100", train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(tv_dataset, lengths=[len(tv_dataset)-int(val_size*len(tv_dataset)), int(val_size*len(tv_dataset))])
    testset = datasets.CIFAR100(root=str(path_to_dir)+"/data/cifar100", train=False, download=True, transform=transform)
    #split into 10 tasks of 10 classes each
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def get_imagenette_dataloaders(path_to_dir="~", val_size=0.1, batch_size=-1, dim=224, woof=False):
    if batch_size < 0:
        batch_size = 32 
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256 if dim == 224 else dim+10),
            transforms.CenterCrop(dim),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    path = str(path_to_dir)+"/data/image" + ("woof" if woof else "nette") + "2-320"
    if not os.path.exists(path):
        print("No data found.")
        return None, None, None
    tv_dataset = datasets.ImageFolder(path+"/train", transform=transform)
    train_set, val_set = torch.utils.data.random_split(tv_dataset, lengths=[len(tv_dataset)-int(val_size*len(tv_dataset)), int(val_size*len(tv_dataset))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(path+"/val", transform=transform), batch_size=batch_size, num_workers=2)
    return train_loader, val_loader, test_loader

def get_imagenet_c_dataloaders(path_to_dir="~", val_size=0.1, test_size=0.2, batch_size=-1, dim=224, corruption="contrast"):
    if batch_size < 0:
        batch_size = 32 
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256 if dim == 224 else dim+10),
            transforms.CenterCrop(dim),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    x, y = load_imagenetc(data_dir=str(path_to_dir)+"/data", corruptions=[corruption], prepr=transform)
    print("Number of samples per class: ", np.unique(y, return_counts=True)[0])
    tvt_dataset = torch.utils.data.TensorDataset(x, y)
    tv_dataset, test_set = torch.utils.data.random_split(tvt_dataset, lengths=[len(tvt_dataset)-int(test_size*len(tvt_dataset)), int(test_size*len(tvt_dataset))])
    train_set, val_set = torch.utils.data.random_split(tv_dataset, lengths=[len(tv_dataset)-int(val_size*len(tv_dataset)), int(val_size*len(tv_dataset))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, val_loader, test_loader


def get_balanced_imagenet_c_dataloaders(path_to_dir="~", val_size=0.2, batch_size=-1, dim=224, corruption="contrast", severity=5):
    if batch_size < 0:
        batch_size = 32 
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256 if dim == 224 else dim+10),
            transforms.CenterCrop(dim),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    tv_path = str(path_to_dir)+"/data/ImageNet-C-train_42/"+corruption+"/"+str(severity)+"/"
    test_path = str(path_to_dir)+"/data/ImageNet-C-test_42/"+corruption+"/"+str(severity)+"/"
    if not os.path.exists(tv_path):
        make_balanced_imagenet_c_folders(path_to_dir=path_to_dir, corruption=corruption, severity=severity)
    tv_dataset = datasets.ImageFolder(tv_path, transform=transform)
    train_set, val_set = torch.utils.data.random_split(tv_dataset, lengths=[len(tv_dataset)-int(val_size*len(tv_dataset)), int(val_size*len(tv_dataset))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(test_path, transform=transform), batch_size=batch_size, num_workers=2)
    return train_loader, val_loader, test_loader

def make_balanced_imagenet_c_folders(path_to_dir="~", trainval_size=-1, test_size=10, corruption="contrast", severity=5):
    print("Making balanced ImageNet-C folders for corruption: "+corruption+" and severity: "+str(severity))
    seed=42
    random.seed(seed)
    if severity == "all":
        origpaths = [str(path_to_dir)+"/data/ImageNet-C/"+corruption+"/"+str(sev)+"/" for sev in range(1,6)]
    else:
        origpaths = [str(path_to_dir)+"/data/ImageNet-C/"+corruption+"/"+str(severity)+"/"]
        trainval_size = 5*trainval_size
    trainpath = str(path_to_dir)+"/data/ImageNet-C-train_"+str(seed)+"/"+corruption+"/"+str(severity)+"/"
    testpath = str(path_to_dir)+"/data/ImageNet-C-test_"+str(seed)+"/"+corruption+"/"+str(severity)+"/"
    os.makedirs(trainpath)
    os.makedirs(testpath)
    for folder in os.listdir(origpaths[-1]):
        if not os.path.exists(folder):
            print("No data found.")
            continue
        os.makedirs(trainpath+folder)
        os.makedirs(testpath+folder)
        #get all images in folder
        images = os.listdir(origpaths[-1]+folder)
        #randomly shuffle
        random.shuffle(images)
        if trainval_size > 0:
            for i in range(trainval_size*len(origpaths),trainval_size*len(origpaths)+test_size):
                shutil.copyfile(origpaths[-1]+folder+"/"+images[i], testpath+folder+"/"+images[i])
            images = images[:trainval_size*len(origpaths)]+images[trainval_size*len(origpaths)+test_size:]
            for i in range(trainval_size):
                shutil.copyfile(origpaths[-1]+folder+"/"+images[i], trainpath+folder+"/"+images[i])
            for origpath in origpaths[:-1]:
                images = [image for image in os.listdir(origpath+folder) if image not in os.listdir(trainpath+folder) and image not in os.listdir(testpath+folder)]
                random.shuffle(images)
                for i in range(trainval_size):
                    shutil.copyfile(origpath+folder+"/"+images[i], trainpath+folder+"/"+images[i])
        else:
            for i in range(test_size):
                shutil.copyfile(origpaths[-1]+folder+"/"+images[i], testpath+folder+"/"+images[i])
            images = images[test_size:]
            for i in range(len(images)):
                shutil.copyfile(origpaths[-1]+folder+"/"+images[i], trainpath+folder+"/"+images[i])
            for origpath in origpaths[:-1]:
                images = [image for image in os.listdir(origpath+folder) if image not in os.listdir(trainpath+folder) and image not in os.listdir(testpath+folder)]
                random.shuffle(images)
                for i in range(len(images)):
                    shutil.copyfile(origpath+folder+"/"+images[i], trainpath+folder+"/"+images[i])