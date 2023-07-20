import torch
from torchvision import transforms
model = torch.load('./AlexNet.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
dummy_input = torch.rand(size=(32,1,224,224),device =device)
print(model(dummy_input))

