import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm 
from model.FeatureExtractor import FeatureExtractor
data_transforms = {
    'train' : transforms.Compose([
        transforms.Resize(64),
        transforms.RandomRotation(120),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ]),
    'test': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])
}
trainData = torchvision.datasets.MNIST('../data_set',download=True,train=True,transform=data_transforms['train'])
testData = torchvision.datasets.MNIST('../data_set',download=True,transform=data_transforms['test'],train=False)

trainDataLoader = DataLoader(trainData,batch_size=64,shuffle=True)
testDataLoader  = DataLoader(testData,batch_size=1024,shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dummy_input  = torch.randn((32,1,64,64),device = device)
model = FeatureExtractor().to(device)
# print(model(dummy_input).shape)
# exit()
model.load_state_dict(torch.load('./FeatureExtractor.pth'))
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
epochs = 10
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
  
torch.save(model.state_dict(),'FeatureExtractor.pth')

