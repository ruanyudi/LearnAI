import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 5)
        )
        print("NN Created !")
        self._init_weight()

    def _init_weight(self):
        for layer in self.modules():
            if (type(layer) == nn.Linear or type(layer) == nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                print(f"init weight : {layer.__class__.__name__}")

    def forward(self, x):
        return self.model(x)


# net = AlexNet().cuda()
# dummy_input = torch.randn(size=(32, 1, 224, 224),device='cuda')
# net(dummy_input)
