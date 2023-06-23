import torch
from torch import nn


class VGGBasicBlock(nn.Module):
    def __init__(self, in_channnels, out_channels, conv_num):
        super().__init__()
        self.block = nn.Sequential()
        for i in range(conv_num):
            self.block.add_module("Conv"+str(i),
                                  module=nn.Conv2d(in_channels=in_channnels, out_channels=out_channels, kernel_size=3,
                                                   padding=1, stride=1))
            self.block.add_module("ReLU"+str(i), nn.ReLU())
            in_channnels = out_channels
        self.block.add_module("MaxPool", nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):
    def __init__(self, in_channels=1, conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        super().__init__()
        self.features = nn.Sequential()
        for i, data in enumerate(conv_arch):
            conv_num,out_channels = data
            self.features.add_module("VGGBasicBlock"+str(i), module=VGGBasicBlock(in_channels, out_channels, conv_num))
            in_channels = out_channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 10)
        )
        self.__init__weight()
        print("Model Loaded Done !")

    def forward(self, x):
        return self.classifier(self.features(x))

    def __init__weight(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
                print(f"init parameters : {layer.__class__.__name__}")

