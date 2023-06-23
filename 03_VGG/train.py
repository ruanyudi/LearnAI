from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from model import VGG

dataTransforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize(224)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224)
    ])
}

trainData = torchvision.datasets.FashionMNIST('../data_set', download=True, transform=dataTransforms['train'],
                                              train=True)
testData = torchvision.datasets.FashionMNIST('../data_set', download=True, transform=dataTransforms['test'],
                                             train=False)
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=32, num_workers=8)
testDataLoader = DataLoader(testData, shuffle=False, batch_size=32, num_workers=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1, momentum=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 10
for epoch in range(epochs):
    train_count = 0
    test_count = 0
    train_loss = 0
    train_acc = 0
    test_acc = 0
    test_loss = 0
    for features, labels in trainDataLoader:
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
    for features, labels in testDataLoader:
        features, labels = features.to(device), labels.to(device)
        pred = model(features)
        loss = loss_fn(pred, labels)
        test_loss += loss.item()
        test_count += len(labels)
        pred = torch.argmax(pred, dim=1)
        test_acc += (pred == labels).sum().item()
    print(
        f"Epoch: {epoch} | TrainLoss: {train_loss / train_count} | TrainACC: {train_acc / train_count} | TestLoss: {test_loss / test_count} | TestACC : {test_acc / test_count}")
torch.save(model, "VGG11.pth")
