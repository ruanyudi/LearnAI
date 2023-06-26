from torch import nn
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from model import VGG
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

#trainData = torchvision.datasets.FashionMNIST('../data_set', download=True, transform=DataTransforms['train'],train=True)
#testData = torchvision.datasets.FashionMNIST('../data_set', download=True, transform=DataTransforms['test'],train=False)
trainData = torchvision.datasets.ImageFolder('../data_set/flower_data/train', transform=DataTransforms['train'])
testData = torchvision.datasets.ImageFolder('../data_set/flower_data/val', transform=DataTransforms['test'])
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=64, num_workers=16)
testDataLoader = DataLoader(testData, shuffle=False, batch_size=1, num_workers=16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = VGG(in_channels=3)
model = torch.load("VGG19.pth")
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 50
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
torch.save(model, "VGG19.pth")
