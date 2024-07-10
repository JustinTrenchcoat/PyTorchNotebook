# Lecture on Multiple dimension dataset?
# Martix multiplication
import torch
import numpy as np
import matplotlib.pyplot as plt

# prepare dataset
# xy is loaded dataset from diabetes.csv. Its delimiter is ',', and its datatype is 32 digit float.
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
# every
x_data = torch.from_numpy(xy[:,:-1])
print('input data shape', x_data.shape)
y_data = torch.from_numpy(xy[:,[-1]])

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

# loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# forward
for epoch in range(1000000):
    y_pred = newModel(x_data)
    loss = criterion(y_pred,y_data)
    # print(epoch,loss.item())as the epoch is too large

# backward
    optimizer.zero_grad()
    loss.backward()
# update
    optimizer.step()
if epoch%100000 == 99999:
    y_pred_label = torch.where(y_pred>=0.5, torch.tensor([1.0]),torch.tensor([0,0]))

    acc = torch.eq(y_pred_label,y_data).sum().item()/y_data.size(0)
    print("loss = ", loss.item(), "acc = ", acc)