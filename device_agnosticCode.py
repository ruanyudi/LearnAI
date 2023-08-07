import torch
def detect_device():
    if torch.cuda.is_available():
        device = 'cuda'
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # GPU's
            templist = [1, 2, 3]
            templist = torch.FloatTensor(templist).to(device)
            print("Cuda torch working: ", end="")
            print(templist.is_cuda)
            print("GPU device count: ", end="")
            print(device_count)
            for i in range(device_count):
                print("GPU name {}: {}".format(i, torch.cuda.get_device_name(i)))
            print("current device no.: ", end="")
            print(torch.cuda.current_device())
        else:
            # A GPU
            print("Cuda torch working: ", end="")
            print(torch.cuda.is_available())
            print("GPU device count: ", end="")
            print(device_count)
            print("GPU name: ", end="")
            print(torch.cuda.get_device_name(0))
            print("current device no.: ", end="")
            print(torch.cuda.current_device())
    else:
        if torch.backends.mps.is_available():
            print("Apple device detected\nActivating Apple Silicon GPU")
            device = torch.device("mps")
        else:
            print("Cannot use GPU, activating CPU")
            device = 'cpu'
    return device