import torch
from tqdm import tqdm

from model import AlexNet
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
DataTransforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}


#trainData = torchvision.datasets.MNIST('../data_set', download=True, transform=dataTransforms['train'], train=True)
#testData = torchvision.datasets.MNIST('../data_set', download=True, transform=dataTransforms['test'], train=False)
trainData = torchvision.datasets.ImageFolder('../data_set/flower_data/train', transform=DataTransforms['train'])
testData = torchvision.datasets.ImageFolder('../data_set/flower_data/val', transform=DataTransforms['test'])
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=512,num_workers=16)
testDataLoader = DataLoader(testData, shuffle=False, batch_size=1,num_workers=16)
epochs = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('./AlexNet.pth').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()


def train():
    for epoch in range(epochs):
        trainTotalLoss = 0
        trainAcc = 0
        trainCount = 0
        testTotalLoss = 0
        testAcc = 0
        testCount = 0
        model.train()
        for i,data in tqdm(enumerate(trainDataLoader)):
            features,labels = data
            features, labels = features.to(device), labels.to(device)
            pred = model(features)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainTotalLoss += loss
            trainCount += len(labels)
            pred = torch.argmax(pred, dim=1)
            trainAcc += (pred == labels).sum().item()
        model.eval()
        with torch.inference_mode():
            for i,data in tqdm(enumerate(testDataLoader)):
                features, labels = data
                features, labels = features.to(device), labels.to(device)
                pred = model(features)
                loss = loss_fn(pred, labels)
                testTotalLoss += loss
                testCount += len(labels)
                pred = torch.argmax(pred, dim=1)
                testAcc += (pred == labels).sum().item()
        print(f"Epoch:{epoch}|TrainLoss:{trainTotalLoss / trainCount:.6f}|TestLoss:{testTotalLoss / testCount:.6f}|trainACC:{trainAcc / trainCount * 100:.2f}%|testACC:{testAcc / testCount * 100:.2f}%")


print("Start to train")
train()
torch.save(model,'AlexNet.pth')
