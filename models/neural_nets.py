from torch import nn
import torch.nn.functional as F
from resnet import ResNet18

###################
###### MNIST ######
###################

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, output = "final"):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if output == "presoft":
            return x
        else:
            x = self.soft(x)
            return x

class MNIST_LeNet(nn.Module):
    def __init__(self):
        super(MNIST_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def encode(self, x):
        x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

    def forward(self, x, output = "final"):
        if output == "presoft":
            return self.encode(x)
        else:
            return F.softmax(self.encode(x), dim=1)


###########################
###### Fashion MNIST ######
###########################

class FashionMNIST_MLP(nn.Module):
    def __init__(self):
        super(FashionMNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, output = "final"):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if output == "presoft":
            return x
        else:
            x = self.soft(x)
            return x

class FashionMNIST_LeNet(nn.Module):
    def __init__(self):
        super(FashionMNIST_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def encode(self, x):
        x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

    def forward(self, x, output = "final"):
        if output == "presoft":
            return self.encode(x)
        else:
            return F.softmax(self.encode(x), dim=1)



###################
###### SVHN #######
###################

class SVHN_LeNet(nn.Module):
    def __init__(self):
        super(SVHN_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, output = "final"):
        x.reshape(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if output == "presoft":
            return x
        else:
            x = F.softmax(x, dim=1)
            return x

###################################
###### SVHN Black and White #######
###################################

class SVHN_LeNet_BandW(nn.Module):
    def __init__(self):
        super(SVHN_LeNet_BandW, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.fc1 = nn.Linear(6 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x, output = "final"):
        x.reshape(-1, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if output == "presoft":
            return x
        else:
            x = F.softmax(x, dim=1)
            return x

###################
###### CIFAR ######
###################

class CIFAR_LeNet(nn.Module):
    def __init__(self):
        super(SVHN_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, output = "final"):
        x.reshape(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if output == "presoft":
            return x
        else:
            x = F.softmax(x, dim=1)
            return x
