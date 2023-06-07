import torch.optim
from torch.utils.data import DataLoader
from model import LeNet
import torchvision
from torchvision import transforms
from vali import vali

DataTransforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
model = LeNet()
batch_size = 32
epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
trainData = torchvision.datasets.CIFAR10('../data_set', download=False, transform=DataTransforms['train'], train=True)
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=batch_size)
testData = torchvision.datasets.CIFAR10('../data_set', download=False, transform=DataTransforms['test'], train=True)
testDataLoader = DataLoader(testData, shuffle=True, batch_size=batch_size)


def train(model,
          epochs,
          optimizer,
          loss_fn,
          trainDataLoader,
          device='cpu'):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for features, labels in trainDataLoader:
            features, labels = features.to(device), labels.to(device)
            pred = model(features)
            loss = loss_fn(pred, labels)
            total_loss, count = total_loss + loss, count + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch : {epoch} | Loss : {total_loss / count:.3f} | TrainAcc : {vali(model,testDataLoader,device)*100:.3f}")


train(model, 10, optimizer, loss_fn, trainDataLoader, device)
torch.save(model.state_dict(), 'LeNet.pth')
