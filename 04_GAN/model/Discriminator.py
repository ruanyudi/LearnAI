from torch import nn
class Discriminator(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        expansion=1
        self.main = nn.Sequential(
            nn.Conv2d(in_channels,64*expansion,kernel_size=3,padding=1,stride=2,bias=False), #32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64*expansion,128*expansion,kernel_size=3,padding=1,stride=2,bias=False),#16
            nn.BatchNorm2d(128*expansion),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128*expansion,256*expansion,kernel_size=3,padding=1,stride=2,bias=False),#8
            nn.BatchNorm2d(256*expansion),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256*expansion,512*expansion,kernel_size=3,padding=1,stride=2,bias=False), #4
            nn.BatchNorm2d(512*expansion),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512*expansion, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).flatten(1)
