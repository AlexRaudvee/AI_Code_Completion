import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader 

batch_size = 64

train_dataset = datasets.MNIST(root = '/Users/alexraudvee/Desktop/TU_e/projects/RNN_with_pytorch/RNN_MNIST/dataset', train=True,
                               transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='/Users/alexraudvee/Desktop/TU_e/projects/RNN_with_pytorch/RNN_MNIST/dataset', train=False, 
                               transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)





