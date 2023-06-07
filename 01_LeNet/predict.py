import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from model import LeNet

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

model = LeNet()
model.load_state_dict(torch.load('./LeNet.pth'))
testData = torchvision.datasets.CIFAR10('../data_set', download=False, transform=DataTransforms['test'], train=False)
testDataLoader = DataLoader(testData, shuffle=True, batch_size=1024)


def vali(model,
         testDataLoader,
         device):
    model = model.to(device)
    model.eval()
    correct = 0
    count = 0
    with torch.inference_mode():
        for features, labels in testDataLoader:
            features, labels = features.to(device), labels.to(device)
            pred = model(features)
            pred = torch.argmax(pred, dim=1)
            correct = correct + (pred == labels).sum().item()
            count = count + len(labels)
    return correct / count


print(f"ACC : {vali(model, testDataLoader, device='cuda')}")
