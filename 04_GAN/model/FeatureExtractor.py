from torch import nn
import torch
class FeatureExtractor(nn.Module):
    #batch_size*1*64*64
    def __init__(self):
        super().__init__()
        self.features =nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1,stride=2), #32
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=2),#16
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2),#8
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2), #4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*256,4096),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.Dropout(),
            nn.Linear(4096,10)
        )
    def forward(self,x):
        return self.classifier(self.features(x))