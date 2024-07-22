# Lecture on convolutional neural network
# mostly used in image processing
# 
import torch
import torch.nn.functional as F

in_channels, out_channels = 5,10
width, height = 100, 100
kernel_size = 3
batch_size = 1

# made a random tensor each dimension represents the image input.
input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)
# a convolutional layer with input channel 5, outputs a channel 10, and the area is 3*3. 
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)
# we could also add padding and stride to manipulate the size of output.

output = conv_layer(input)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(320,10)
    def forward(self,x):
        #flattens data from (n,1,28,28) to (n,784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))


