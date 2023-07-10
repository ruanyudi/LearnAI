import torchvision.datasets
from tqdm import tqdm
from torchvision import transforms
from torch import nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.FeatureExtractor import FeatureExtractor
from model.Discriminator import Discriminator
from model.Generator import Generator
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeatureExtractor(FeatureExtractor):
    def forward(self,x):
        return self.features(x)
        

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomRotation(180),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]),
    'flower': transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
}


def show_img(model):
    fig = plt.figure(figsize=(10, 10))
    noise_input = torch.randn((100, 100), device=device)
    prediction = model(noise_input).detach().cpu()
    prediction = torch.transpose(prediction,1,3).numpy()
    
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow((prediction[i] + 1) / 2, cmap='gray')
        plt.axis('off')
    plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def loss_optical_fn(x,y):
    # batch_size*256*4*4
    x,y = x.flatten(0),y.flatten(0)
    return (torch.abs(x-y)).mean()

def train(epochs,lr):
    # trainData = torchvision.datasets.CIFAR10('../data_set', download=True, transform=data_transforms['train'], train=True)
    trainData = torchvision.datasets.MNIST('../data_set', download=True, transform=data_transforms['train'], train=False)
    # trainData = torchvision.datasets.ImageFolder('../data_set/flower_data/flower_photos',transform = data_transforms['flower'])
    trainDataLoader = DataLoader(trainData, batch_size=128, shuffle=False,num_workers=16)
    Gen = Generator(out_channels=1).to(device)
    Gen.apply(weights_init)
    Dis = Discriminator(in_channels=1).to(device)
    Dis.apply(weights_init)
    featureExtractor = FeatureExtractor().to(device)
    featureExtractor.load_state_dict(torch.load('FeatureExtractor.pth'))
    featureExtractor.eval()
    # try:
    #     Gen = torch.load('Gen.pth')
    #     Dis = torch.load('Dis.pth')
    # except:
    #     print("Fail to Loader the Weight")
    # print(featureExtractor(dummy_input).shape)
    G_optimizer = torch.optim.Adam(Gen.parameters(), lr=lr,betas=(0.5,0.999))
    D_optimizer = torch.optim.Adam(Dis.parameters(), lr=lr,betas=(0.5,0.999))
    loss_fn = torch.nn.BCELoss()
    for epoch in range(epochs):
        G_loss = 0.
        D_loss_GT = 0.
        D_loss_GE = 0.
        D_loss = 0.
        count = len(trainDataLoader)
        for step, data in tqdm(enumerate(trainDataLoader)):
            features, _ = data
            noise_input = torch.randn((len(features), 100), device=device)
            gen_result = Gen(noise_input)
            # train the discriminator
            features = features.to(device)
            Dis.zero_grad()
            pred_gt = Dis(features)
            pred_ge = Dis(gen_result.detach())
            loss_gt = loss_fn(pred_gt, torch.ones_like(pred_gt, device=device))
            loss_ge = loss_fn(pred_ge, torch.zeros_like(pred_ge, device=device))
            loss_D = loss_gt + loss_ge
            D_loss_GE +=loss_ge
            D_loss_GT +=loss_gt
            D_loss += loss_D
            D_optimizer.zero_grad()
            loss_D.backward()
            D_optimizer.step()

            loss_G = loss_fn(Dis(gen_result), torch.ones((len(features),1), device=device))
            G_loss += loss_G
            G_optimizer.zero_grad()
            loss_G.backward()
            G_optimizer.step()
        print(f"Epoch :{epoch} | D_Loss :{D_loss / count} | G_Loss : {G_loss / count} | GT :{D_loss_GT/count} | GE :{D_loss_GE/count}")
        if(epoch%5==0):
            show_img(Gen)
        torch.save(Gen, 'Gen.pth')
        torch.save(Dis, 'Dis.pth')

