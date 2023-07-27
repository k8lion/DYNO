# DYNO: Dynamic Neural Optimization

This repository contains the code for the paper:

"Neural Growth and Pruning in Dynamic Learning Environments." Kaitlin Maile, Hervé Luga, and Dennis G. Wilson. Second Conference on Automated Machine Learning (Workshop Track). 2023.


## Requirements

Make and activate a clean conda or virtual envirnoment. Clone [NeurOps](https://github.com/SuReLI/NeurOps) and add it to your `PYTHONPATH`. Then, run the following command from the root directory of this repository:

```
python3 -m pip install -r requirements.txt
```

The following datasets are used in this repository. Follow each link to documentation on how to download the dataset. Place the datasets in a `data` folder, which by default is assumed to be in the same directory as this repository but can be specified elsewise using the `--path` argument.
- [ImageWoof/ImageNette](https://github.com/fastai/imagenette)
- [Galaxy10](https://astronn.readthedocs.io/en/latest/galaxy10.html)
- [Corrupted Imagenet](https://github.com/hendrycks/robustness)

The following pretrained models are used in this repository. Each link will directly download the model weights. Place these files in the same `data` folder as the datasets.
- [Imagenet-pretrained VGG11](https://download.pytorch.org/models/vgg11-8a719046.pth)
- [Imagenet-pretrained VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)

## Running Experiments

To run an experiment, use the following command:

```
python3 src/transfer.py
```

See the file `src/transfer.py` for the full list of arguments.