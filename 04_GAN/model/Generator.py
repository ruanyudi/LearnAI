from torch import nn
class Generator(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        expansion=1
        self.DeConv = nn.Sequential(
            nn.ConvTranspose2d(100,1024*expansion,kernel_size=4,padding=0,stride=1,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1024*expansion, 512*expansion, kernel_size=4, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(512*expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(512*expansion, 256*expansion, kernel_size=4, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(256*expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(256*expansion, 128*expansion, kernel_size=4, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(128*expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(128*expansion, out_channels, kernel_size=4, padding=1, stride=2,bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(-1,100,1,1)
        return self.DeConv(x)