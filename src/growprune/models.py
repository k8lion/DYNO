import sys
import os
sys.path.append(os.path.expanduser("~/repos/NeurOps/pytorch"))
from neurops import *
import torch


class ModMLP(ModSequential):
    def __init__(self, inputs, hidden, outputs: int, track_activations=False, norm_weights=False):
        super().__init__(
            ModLinear(inputs, hidden, masked=True),
            ModLinear(hidden, outputs, masked=True, nonlinearity=''),
            track_activations=track_activations,
        )
        self.num_classes = outputs


    def forward(self, x, layer_index=-1):
        if self.num_classes == 1:
            return super().forward(x).flatten()
        return super().forward(x)


class ModVGG11(ModSequential):
    def __init__(self, num_classes: int = 10, avgpooldim = 1, debug=False):
        super().__init__(
            ModConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, 
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ModConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                      postpool=torch.nn.Sequential(torch.nn.MaxPool2d(2,2), 
                                                   torch.nn.AdaptiveAvgPool2d((avgpooldim, avgpooldim)))), 
            ModLinear(512*avgpooldim**2, 512 if debug else 4096, preflatten=True),
            ModLinear(512 if debug else 4096, 512 if debug else 4096),
            ModLinear(512 if debug else 4096, num_classes, nonlinearity=''),
            track_activations=True,
        )
        self.conversion_factor = avgpooldim**2
        self.num_classes = num_classes
        self.weights_file = "vgg11-8a719046.pth"



class ModVGG19(ModSequential):
    def __init__(self, num_classes: int = 10, avgpooldim = 1, debug=False):
        super().__init__(
            ModConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            ModConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, 
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ModConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ModConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            ModConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            ModConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                      postpool=torch.nn.Sequential(torch.nn.MaxPool2d(2,2), 
                                                   torch.nn.AdaptiveAvgPool2d((avgpooldim, avgpooldim)))), 
            ModLinear(512*avgpooldim**2, 512 if debug else 4096, preflatten=True),
            ModLinear(512 if debug else 4096, 512 if debug else 4096),
            ModLinear(512 if debug else 4096, num_classes, nonlinearity=''),
            track_activations=True,
        )
        self.conversion_factor = avgpooldim**2
        self.num_classes = num_classes
        self.weights_file = "vgg19-dcbb9e9d.pth"
