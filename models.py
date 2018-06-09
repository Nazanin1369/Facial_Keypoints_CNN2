## TODO: define the convolutional neural network architecture
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 5x5 square convolution kernel
        
        #self.pool = nn.MaxPool2d(2,2)
            
        ## output size = (W-F)/S +1 = (224 - 5)/1 +1 = 220
        # after conv 1, the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout2d(p= 0.1)

        # second conv layer: 32 inputs, 48 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110 - 5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53);
        self.conv2 = nn.Conv2d(32, 48, 5)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv2_drop = nn.Dropout2d(p= 0.1)

        # third conv layer: 64 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (53 - 5)/1 +1 = 49
        # the output tensor will have dimensions: (64, 49, 49)
        # after another pool layer this becomes (64, 24, 24);
        self.conv3 = nn.Conv2d(48, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_drop = nn.Dropout2d(p = 0.2)

        # fourth conv layer: 64 inputs, 96 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (24 - 5)/1 +1 = 20
        # the output tensor will have dimensions: (96, 20, 20)
        # after another pool layer this becomes (96, 10, 10);
        self.conv4 = nn.Conv2d(64, 96, 5)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.conv4_drop = nn.Dropout2d(p = 0.2)

        self.fc1 = nn.Linear(96*10*10, 96*5*5)
        self.fc1_bn = nn.BatchNorm1d(96*5*5)
        
        # dropout with p=0.2
        self.fc1_drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(96*5*5, 68*3*3)
        self.fc2_bn = nn.BatchNorm1d(68*3*3)

        # dropout with p=0.3
        self.fc2_drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(68*3*3, 68*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)),2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)),2))
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)),2))
        x = self.conv3_drop(x)
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)),2))
        x = self.conv4_drop(x)

        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # fc 1
        x = self.fc1_drop(F.relu(self.fc1_bn(self.fc1(x))))
        
        # fc 2
        x = self.fc2_drop(F.relu(self.fc2_bn(self.fc2(x))))
        
        # fc 3
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    