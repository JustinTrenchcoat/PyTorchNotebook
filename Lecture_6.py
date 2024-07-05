# Lecture on Multiple dimension dataset?
# Martix multiplication
import torch
import numpy as np

#we need to define this batchSize
global batchSize

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # input feature is batchSize, output feature is 1 (# of columns?)
        # essentailly torch.nn.Linear linearly(no shit) maps a matrix of (batchSize)\
        # dimension to matrix of dimension 1, or really any dimension we define.
        self.linear = torch.nn.Linear(batchSize,1)
        # use Sigmoid to somehow simulate a non-linear transformation.
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear(x))
        return x

model = Model()


# to make a more complex neural network, we could 'stack' them together
class stackModel(torch.nn.Module):
    def __init__(self):
        super(stackModel,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
newModel = stackModel()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

# forward
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

# backward
    optimizer.zero_grad()
    loss.backward()
# update
    optimizer.step()

