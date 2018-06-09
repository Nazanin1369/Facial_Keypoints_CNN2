## Define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 32, 5)
        self.conv1_2 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.4)
        
        
        self.conv2_1 = nn.Conv2d(1, 32, 5)
        self.conv2_2 = nn.Conv2d(1, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.4)
        
        
        self.conv3_1 = nn.Conv2d(1, 32, 5)
        self.conv3_2 = nn.Conv2d(1, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.4)
        
        self.conv4_1 = nn.Conv2d(1, 32, 5)
        self.conv4_2 = nn.Conv2d(1, 32, 5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(128*24*24, 1000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = self.droupout1(x)
        
        
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = self.droupout2(x)
        
        x = F.relu(self.con3_1(x))
        x = self.pool3(F.relu(self.conv3_2(x)))
        x = self.droupout3(x)
        
        
        x = F.relu(self.conv4_1(x))
        x = self.pool4(F.relu(self.conv4_2(x)))
        x = self.droupout4(x)
        
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

