import torch
import torchvision.datasets
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VGG

dataTransforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(180),
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
}

trainData = torchvision.datasets.MNIST('../data_set',download=True,train=True,transform=dataTransforms['train'])
testData = torchvision.datasets.MNIST('../data_set',download=True,train=False,transform=dataTransforms['test'])
trainDataLoader = DataLoader(trainData,shuffle=False,batch_size=64,num_workers=16)
testDataLoader = DataLoader(testData,shuffle=False,batch_size=1,num_workers=16)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 80
for epoch in range(epochs):
    train_count = 0
    test_count = 0
    train_loss = 0
    train_acc = 0
    test_acc = 0
    test_loss = 0
    model.train()
    for i,data in tqdm(enumerate(trainDataLoader)):
        features,labels = data
        features, labels = features.to(device), labels.to(device)
        pred = model(features)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_count += len(labels)
        pred = torch.argmax(pred, dim=1)
        train_acc += (pred == labels).sum().item()
    model.eval()
    for i,data in tqdm(enumerate(testDataLoader)):
        features,labels = data
        features, labels = features.to(device), labels.to(device)
        pred = model(features)
        loss = loss_fn(pred, labels)
        test_loss += loss.item()
        test_count += len(labels)
        pred = torch.argmax(pred, dim=1)
        test_acc += (pred == labels).sum().item()
    print(
        f"Epoch: {epoch} | TrainLoss: {train_loss / train_count} | TrainACC: {train_acc / train_count*100:.2f} | TestLoss: {test_loss / test_count} | TestACC : {test_acc / test_count*100:.2f}")
