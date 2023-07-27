import torch
from collections import OrderedDict
import argparse
import numpy as np
import pickle
import copy
import math
from neurops import *
from growprune import *

g10small_thresholds=[1.0]*10
g10small_thresholds[-1]=4066.0/4096
g10_thresholds=[1.0]*10
g10_thresholds[-1]=4086.0/4096
woof2net = [155, 159, 162, 167, 182, 193, 207, 229, 258, 273]
nette2net = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

def transfer(args):
    if "static" in args.prune_score:
        args.no_prune = True
        args.no_grow = True
    if "surgical" in args.prune_score:
        args.surgical = True
    if "reinitdense" in args.prune_score:
        args.reset_linear = True
    if args.pruning_threshold == -1:
        args.pruning_threshold = 0.8 if args.old else 0.97
    if args.growing_threshold == -1:
        args.growing_threshold = 0.98 if args.old else 1.03
    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
    filename = str(args.path) +'/out/'+args.task+"_"+args.name+'.pkl'
    if not os.path.isdir(args.path+"/out"):
        os.mkdir(args.path+"/out")
    print(filename)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if "galaxy10" in args.task:
        if args.vggnineteen:
            model = ModVGG19(num_classes=10, avgpooldim=1 if "small" in args.task else 7).to(device)
        else:
            model = ModVGG11(num_classes=10, avgpooldim=1 if "small" in args.task else 7).to(device)
        limit = len(model.state_dict())-4 if args.reset_linear or "small" in args.task else len(model.state_dict())
        if not args.no_pretrain:
            weights = torch.load(args.path+"/data/"+model.weights_file)
            renamed = OrderedDict()
            for i, (key, value) in enumerate(zip(model.state_dict(), weights.values())):
                if i <= limit and value.shape == model.state_dict()[key].shape:
                    print("loading pretrained weights for "+key)
                    renamed[key] = value
                else:
                    print("not loading weights for "+key)
                    renamed[key] = model.state_dict()[key]
            model.load_state_dict(renamed)
        criterion = torch.nn.CrossEntropyLoss()
        input = torch.zeros(1,3,64 if "small" in args.task else 224,64 if "small" in args.task else 224)
        train_loader, validation_loader, test_loader = get_galaxy10_dataloaders(path_to_dir=args.path, batch_size=args.batch_size, 
                                                                                dim=64 if "small" in args.task else 224, random_seed=args.seed)
        if args.recompute_stats:
            validation_loader_ = get_imagenet_dataloader(path_to_dir=args.path, batch_size=args.batch_size,
                                                         dim=64 if "small" in args.task else 224)
    elif "imagenet" in args.task or "imagewoof" in args.task:
        if args.vggnineteen:
            model = ModVGG19(num_classes=10 if ("woof" in args.task or "nette" in args.task) else 1000, avgpooldim=1 if "small" in args.task else 7).to(device)
        else:
            model = ModVGG11(num_classes=10 if ("woof" in args.task or "nette" in args.task) else 1000, avgpooldim=1 if "small" in args.task else 7).to(device)
        limit = len(model.state_dict())-4 if args.reset_linear or "small" in args.task else len(model.state_dict())
        if not args.no_pretrain:
            weights = torch.load(args.path+"/data/"+model.weights_file)
            renamed = OrderedDict()
            for i, (key, value) in enumerate(zip(model.state_dict(), weights.values())):
                if i <= limit and value.shape == model.state_dict()[key].shape:
                    print("loading pretrained weights for "+key)
                    renamed[key] = value
                elif not args.reset_linear and "reinit" not in args.prune_score and("woof" in args.task or "nette" in args.task):
                    print("loading sliced pretrained weights for "+key)
                    indices = woof2net if "imagewoof" in args.task else nette2net
                    renamed[key] = value[indices]
                else:
                    print("not loading weights for "+key)
                    renamed[key] = model.state_dict()[key]
            model.load_state_dict(renamed)
        criterion = torch.nn.CrossEntropyLoss()
        input = torch.zeros(1,3,64 if "small" in args.task else 224,64 if "small" in args.task else 224)
        if "imagenette" in args.task:
            train_loader, validation_loader, test_loader = get_imagenette_dataloaders(path_to_dir=args.path, batch_size=args.batch_size, woof=False,
                                                                                    dim=64 if "small" in args.task else 224)
        elif "imagewoof" in args.task:
            train_loader, validation_loader, test_loader = get_imagenette_dataloaders(path_to_dir=args.path, batch_size=args.batch_size, woof=True,
                                                                                    dim=64 if "small" in args.task else 224)
        elif "imagenet_c" in args.task:
            train_loader, validation_loader, test_loader = get_balanced_imagenet_c_dataloaders(path_to_dir=args.path, batch_size=args.batch_size,
                                                                                    dim=64 if "small" in args.task else 224, corruption=args.corruption, 
                                                                                    severity=args.severity)
            print("Using balanced Imagenet-C dataloader with corruption "+args.corruption+" and severity"+str(args.severity))
            print("Number of training samples: "+str(len(train_loader.dataset)))
            print("Number of validation samples: "+str(len(validation_loader.dataset)))
            print("Number of test samples: "+str(len(test_loader.dataset)))
        else:
            if "imagenetval" not in args.task:
                print("Task not recognized. Defaulting to imagenetval.")
            validation_loader = get_imagenet_dataloader(path_to_dir=args.path, batch_size=args.batch_size,
                                                            dim=64 if "small" in args.task else 224)
            train_loader = validation_loader
            test_loader = validation_loader
        if args.recompute_stats:
            if "imagenetval" in args.task:
                validation_loader_ = validation_loader
            else:
                validation_loader_ = get_imagenet_dataloader(path_to_dir=args.path, batch_size=args.batch_size,
                                                            dim=64 if "small" in args.task else 224)
    input = input.to(device)
    if args.surgical:
        optimizer = torch.optim.Adam([{'params': model[layer].parameters(), 'lr': args.learning_rate} for layer in range(len(model))], lr=args.learning_rate, weight_decay=args.weight_decay)
        layer_weights = [0 for layer, _ in model.named_parameters() if 'bn' not in layer]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = {'args': args,
               'widths': {str(layer): [model[layer].width(masked="toy" in args.task)] for layer in range(len(model))},
               'pruned': {str(layer): [] for layer in range(len(model))},
               'grown': {str(layer): [] for layer in range(len(model))},
               'scores': {str(layer): [] for layer in range(len(model))},
               'neuron_scores': {str(layer): [] for layer in range(len(model))},
               'svd_scores' : {str(layer): np.zeros((0,model[layer].width())) for layer in range(len(model))},
               'corr_scores' : {str(layer): np.zeros((0,model[layer].width()))  for layer in range(len(model))},
               'actvar_scores' : {str(layer): np.zeros((0,model[layer].width()))  for layer in range(len(model))},
               'weight_scores' : {str(layer): np.zeros((0,model[layer].width()))  for layer in range(len(model))},
               'apoz_scores' : {str(layer): np.zeros((0,model[layer].width()))  for layer in range(len(model))},
               'dropcorr_scores' : {str(layer): np.zeros((0,model[layer].width()))  for layer in range(len(model))},
               'recon_scores' : {str(layer): np.zeros((0,model[layer].width()))  for layer in range(len(model))},
               'train': {'loss': [], 
                         'accuracy': [], 
                         'batchloss': []},
               'widthsteps': [0],
               'pre-steps': [],
               'pre-train validation' : {'loss': [], 
                                         'accuracy': []},
               'post-train validation' : {'loss': [], 
                                         'accuracy': []},
               'paramcount': [model.parameter_count()], 
               'flops': [model.FLOPs_count(input, verbose=True)],
    }
    if args.recompute_stats and validation_loader_ is not None:
        test(model, validation_loader_, criterion, device=device, act_only=True)
    threshold_score = np.zeros(len(model))
    for i in range(len(model)-1):
        if args.recompute_stats or "toy" in args.task:
            threshold_score[i] = effective_rank(model.activations[str(i)], limit_ratio = 10)/model[i].width(masked="toy" in args.task)
        elif "galaxy10" in args.task:
            threshold_score[i] = g10small_thresholds[i] if "small" in args.task else g10_thresholds[i]
        else:
            threshold_score[i] = 1.0
        width=model[i].width(masked="toy" in args.task)
        print(f"Layer {i} has effective rank {int(threshold_score[i]*width)}/{width} on previous distribution")
        history["scores"][str(i)].append(int(threshold_score[i]*model[i].width(masked="toy" in args.task)))
    iterations = 0
    arch_changed = True
    if train_loader is None and validation_loader is None:
        return -1
    print("begin transfer")
    cycles = int(args.total_iters/args.gp_iters)
    if not args.train_first and train_loader is not None:
        cycles = cycles + 1 
        test(model, train_loader, criterion, device=device, act_only=True)
    for gpcycle in range(cycles):
        if arch_changed:
            if validation_loader is not None:
                loss, acc = test(model, validation_loader, criterion, device=device)
                for i in range(len(model)-1):
                    if threshold_score[i] < 0:
                        threshold_score[i] =  effective_rank(model.activations[str(i)], limit_ratio = 3 if "toy" not in args.task else -1)/model[i].width(masked="toy" in args.task)
                history['pre-train validation']['loss'].append(loss)
                history['pre-train validation']['accuracy'].append(acc)
        else:
            loss = history['post-train validation']['loss'][-1]
            acc = history['post-train validation']['accuracy'][-1]
            history['pre-train validation']['loss'].append(loss)
            history['pre-train validation']['accuracy'].append(acc)
        if not args.train_first and gpcycle == 0:
            if validation_loader is not None:
                val_losses, val_accs, _ = train(model, train_loader, optimizer, criterion, iterations=0, val_loader=validation_loader, device=device, verbose=True)
                history['post-train validation']['loss'].extend(val_losses)
                history['post-train validation']['accuracy'].extend(val_accs)
        else:
            if train_loader is not None:
                if args.surgical:
                    optimizer = perform_surgery(optimizer, model, train_loader, criterion, layer_weights, device)
                val_losses, val_accs, _ = train(model, train_loader, optimizer, criterion, iterations=args.gp_iters, val_loader=validation_loader, device=device, verbose=True)
                iterations += args.gp_iters
                history['post-train validation']['loss'].extend(val_losses)
                history['post-train validation']['accuracy'].extend(val_accs)
        arch_changed = False
        for i in range(len(model)-1):  
            max_rank = model[i].width(masked="toy" in args.task)
            score = effective_rank(model.activations[str(i)], limit_ratio = 3 if "toy" not in args.task else -1)
            print("Layer {} width: {}, score: {}<{:.3f}?".format(i, max_rank, score.item(), args.pruning_threshold*threshold_score[i]*max_rank), end=" ")
            if (args.no_grow and "toy" in args.task):
                svd_scores = svd_score(model.activations[str(i)], limit_ratio = 3 if "toy" not in args.task else -1)
                corr_scores = correlation_score(model.activations[str(i)])
                actvar_scores = activation_variance(model.activations[str(i)])
                weight_scores = weight_sum(model[i].weight, p=2)
                apoz_scores = apoz_score(model.activations[str(i)])
                dropcorr_scores = dropped_corr_score(model.activations[str(i)])
                recon_scores = reconstruction_score(model.activations[str(i)], limit_ratio = 3 if "toy" not in args.task else -1)
                history['svd_scores'][str(i)] = np.concatenate((history['svd_scores'][str(i)], np.expand_dims(svd_scores.cpu().detach().numpy(), 0)), axis=0)
                history['corr_scores'][str(i)] = np.concatenate((history['corr_scores'][str(i)], np.expand_dims(corr_scores.cpu().detach().numpy(), 0)), axis=0)
                history['actvar_scores'][str(i)] = np.concatenate((history['actvar_scores'][str(i)], np.expand_dims(actvar_scores.cpu().detach().numpy(), 0)), axis=0)
                history['weight_scores'][str(i)] = np.concatenate((history['weight_scores'][str(i)], np.expand_dims(weight_scores.cpu().detach().numpy(), 0)), axis=0)
                history['apoz_scores'][str(i)] = np.concatenate((history['apoz_scores'][str(i)], np.expand_dims(apoz_scores.cpu().detach().numpy(), 0)), axis=0)
                history['dropcorr_scores'][str(i)] = np.concatenate((history['dropcorr_scores'][str(i)], np.expand_dims(dropcorr_scores.cpu().detach().numpy(), 0)), axis=0)
                history['recon_scores'][str(i)] = np.concatenate((history['recon_scores'][str(i)], np.expand_dims(recon_scores.cpu().detach().numpy(), 0)), axis=0)
            num_to_prune, to_add = 0, 0
            if not args.no_prune:
                #num_to_prune = min(max(int(args.pruning_threshold*threshold_score[i]*max_rank-score), 0), int(0.5*max_rank))
                num_to_prune = max(int(args.pruning_threshold*threshold_score[i]*max_rank-score), 0)
                if isinstance(model[i], nn.Linear) and max_rank - num_to_prune < model.num_classes:
                    num_to_prune = max_rank - model.num_classes
                to_prune = []
                if num_to_prune > 0 or "toy" in args.task or args.prune_score == "imp":
                    if args.prune_score == "apoz":
                        scores = apoz_score(model.activations[str(i)])
                        to_prune = np.argsort(scores.cpu().detach().numpy())[-num_to_prune:]
                    elif args.prune_score == "actvar":
                        scores = activation_variance(model.activations[str(i)])
                        to_prune = np.argsort(scores.cpu().detach().numpy())[:num_to_prune]
                    elif args.prune_score == "recon":
                        scores = reconstruction_score(model.activations[str(i)], limit_ratio = 3 if "toy" not in args.task else -1)
                        to_prune = np.argsort(scores.cpu().detach().numpy())[:num_to_prune]
                    elif "corr_greedy" in args.prune_score:
                        to_prune = corr_greedy_prune(model.activations[str(i)], num_to_prune, device=device)
                    else:
                        scores = weight_sum(model[i].weight, p=2)
                        to_prune = np.argsort(scores.cpu().detach().numpy())[:num_to_prune]

                    if "toy" in args.task:
                        to_prune = [neuron for neuron in to_prune if model[i].mask_vector[neuron] > 0]

                    if "corr_greedy" not in args.prune_score:
                        history['neuron_scores'][str(i)].append(scores.cpu().detach().numpy())

                    if "toy" in args.task:
                        model.mask(i, to_prune)
                        print("Masked neurons {}".format(to_prune), end=" ")
                    else:
                        before = model[i].width(masked="toy" in args.task)
                        model.prune(i, to_prune, optimizer=optimizer)
                        print("Pruned {} neurons".format(before-model[i].width(masked="toy" in args.task)), end=" ")
                    arch_changed = True
                    score = effective_rank(model.activations[str(i)], limit_ratio = 3 if "toy" not in args.task else -1)
                history['pruned'][str(i)].append(num_to_prune)
            if not args.no_grow:
                grow_threshold = args.growing_threshold*threshold_score[i] #min(args.growing_threshold*threshold_score[i], 0.95)
                print("score: {}>{:.3f}?".format(score.item(), grow_threshold*max_rank), end=" ")
                to_add = max(int(score-grow_threshold*max_rank), 0)
                print("# neurons to add: {}".format(to_add), end=" ")
                if to_add > 0:
                    model.grow(i, to_add, fanin_weights="north_select", 
                                    optimizer=optimizer)
                    arch_changed = True
                history['grown'][str(i)].append(to_add)
            if not args.old and (num_to_prune > 0 or to_add > 0):
                threshold_score[i] = -1
            print()
            history['widths'][str(i)].append(int(model[i].width(masked="toy" in args.task)))
            history['scores'][str(i)].append(score.item())
            if args.old and score > threshold_score[i]*max_rank:
                threshold_score[i] = score/max_rank
        history['paramcount'].append(model.parameter_count())
        history['flops'].append(model.FLOPs_count(input))
        history['widthsteps'].append(iterations)
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
    model.FLOPs_count(input, verbose=True)
    if test_loader is not None:
        _, test_acc = test(model, test_loader, criterion, verbose=False, device=device)
        history['test'] = test_acc
    with open(filename, 'wb') as f:
        pickle.dump(history, f)
    return history['post-train validation']['accuracy'][-1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run grow and prune')
    parser.add_argument('--task', "-t", type=str, default="galaxy10", help='task')
    parser.add_argument('--reset_linear', action='store_true', default=False, help='start linear layers w/ random init')
    parser.add_argument('--pruning_threshold', type=float, default=-1, help='pruning threshold')
    parser.add_argument('--growing_threshold', type=float, default=-1, help='growing threshold')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--total_iters', type=int, default=1000, help='total number of iterations')
    parser.add_argument('--gp_iters', type=float, default=50, help='number of iterations between growing/pruning')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--path', type=str, default="..", help="path for storing data and outputs")
    parser.add_argument('--name', "-n", type=str, default="test", help='name of experiment')
    parser.add_argument('--no_grow', action='store_true', default=False, help='turn off growing')
    parser.add_argument('--no_prune', action='store_true', default=False, help='turn off pruning')
    parser.add_argument('--no_pretrain', action='store_true', default=False, help='turn off pretraining')
    parser.add_argument('--prune_score', type=str, default="corr_greedy", help="pruning strategy")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--recompute_stats', action='store_true', default=False, help='recompute dataset statistics')
    parser.add_argument('--old', action='store_true', default=False, help='use old NORTH* scheduling')
    parser.add_argument('--train_first', action='store_true', default=False, help='do cycle of training before g&p')
    parser.add_argument('--surgical', action='store_true', default=False, help='use surgical fine-tuning (Lee & Chen et al., 2023)')
    parser.add_argument('--corruption', type=str, default="contrast", help='type of corruption for imagenet-c')
    parser.add_argument('--severity', type=str, default="1", help='level of corruption for imagenet-c')
    parser.add_argument('--vggnineteen', action='store_true', default=False, help='use vgg19 instead of vgg11')
    transfer(parser.parse_args())