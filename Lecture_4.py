# Lecture on Linear Regression
# we shall use mini-batch fashion to train

import torch

# x and y are 3*1 size matrix
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# we inherit from nn.Module, 
# which is the base class for all neural networks 
class LinearModel(torch.nn.Module):
    # to keep in mind, __init__() and forward() have to be implemented.
    def __init__(self):
        # just do it. It calls the __init__() of its parent(?)
        super(LinearModel,self).__init__()
        #torch.nn.Linear(1,1) creates an object named linear 
        # with weight 1, bias 1: in y = w*x+b, the w and b. 
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinearModel()