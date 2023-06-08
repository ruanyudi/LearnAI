from model import LeNet
import torchvision
from torchvision import transforms
import torch.optim
from torch.utils.data import DataLoader

# DataTransforms = {
#     'train': transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]),
#     'test': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]),
# }
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = LeNet()
# batch_size = 32
# loss_fn = torch.nn.CrossEntropyLoss()
# testData = torchvision.datasets.CIFAR10('../data_set', download=True, transform=DataTransforms['test'], train=True)
# testDataLoader = DataLoader(testData, shuffle=True, batch_size=batch_size)

def vali(model,
         testDataLoader,
         device):
    model = model.to(device)
    model.eval()
    correct = 0
    count =0
    with torch.inference_mode():
        for features,labels in testDataLoader:
            features,labels = features.to(device),labels.to(device)
            pred = model(features)
            pred = torch.argmax(pred,dim=1)
            correct = correct + (pred==labels).sum().item()
            count = count + len(labels)
    return correct/count