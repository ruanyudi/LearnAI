import torch.optim
from torch.utils.data import DataLoader
from model import LeNet
import torchvision
from torchvision import transforms
from vali import vali
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


class FlowerCF(nn.Module):
    def __init__(self):
        super().__init__()
        self.preNet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=5, stride=2, padding=2),  # 112
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2),  # 56
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, stride=2, padding=4),  # 28
            nn.BatchNorm2d(2048),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=1)
        )
        self.LeNet = LeNet()

    def forward(self, x):
        return self.LeNet(self.preNet(x))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
model = FlowerCF()
batch_size = 32
epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
trainData = torchvision.datasets.ImageFolder('../data_set/flower_data/train', transform=DataTransforms['train'])
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=batch_size)
testData = torchvision.datasets.ImageFolder('../data_set/flower_data/val', transform=DataTransforms['test'])
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
        print(
            f"Epoch : {epoch} | Loss : {total_loss / count:.3f} | TrainAcc : {vali(model, testDataLoader, device) * 100:.2f}%")


print(f"Start to Train the Model...")
train(model, 50, optimizer, loss_fn, trainDataLoader, device)
# torch.save(model.state_dict(), 'LeNet.pth')
