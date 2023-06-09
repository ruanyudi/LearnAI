import numpy as np
import torch.optim
from prettytable import PrettyTable
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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),  # 112
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),  # 56
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=6),  # 32
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1)
        )
        self.LeNet = LeNet()

    def forward(self, x):
        x = self.preNet(x)
        return self.LeNet(x)


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
            f"Epoch : {epoch} | Loss : {total_loss / count:.3f} | Val Acc : {vali(model, testDataLoader, device) * 100:.2f}%")


print(f"Start to Train the Model...")
try:
    model.load_state_dict(torch.load('LeNet_flower.pth'))
except Exception:
    print("Failed to Load the Weight")


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)


confusionmatrix = ConfusionMatrix(5, trainData.classes)

train(model, 1, optimizer, loss_fn, trainDataLoader, device)
vali(model,testDataLoader,device,confusionmatrix)
confusionmatrix.summary()
torch.save(model.state_dict(), 'LeNet_flower.pth')
