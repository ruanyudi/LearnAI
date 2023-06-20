import torch
from model import AlexNet
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn

dataTransforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
}

trainData = torchvision.datasets.MNIST('../data_set', download=True, transform=dataTransforms['train'], train=True)
testData = torchvision.datasets.MNIST('../data_set', download=True, transform=dataTransforms['test'], train=False)
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=1)
testDataLoader = DataLoader(testData, shuffle=False, batch_size=32)
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
        for features, labels in trainDataLoader:
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
            for features, labels in testDataLoader:
                features, labels = features.to(device), labels.to(device)
                pred = model(features)
                loss = loss_fn(pred, labels)
                testTotalLoss += loss
                testCount += len(labels)
                pred = torch.argmax(pred, dim=1)
                testAcc += (pred == labels).sum().item()
        print(
            f"Epoch:{epoch}|TrainLoss:{trainTotalLoss / trainCount:.6f}|TestLoss:{testTotalLoss / testCount:.6f}|trainACC:{trainAcc / trainCount * 100:.2f}%|testACC:{testAcc / testCount * 100:.2f}%")


print("Start to train")
train()
