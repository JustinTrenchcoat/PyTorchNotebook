# Lecture on Linear Regression
# we shall use mini-batch fashion to train

import torch

# x and y are 3*1 size matrix
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# we inherit from nn.Module, 
# which is the base class for all neural networks 
# therefore,, LinearModel is a vanilla neural network with our own definition
# 
class LinearModel(torch.nn.Module):
    # to keep in mind, __init__() and forward() have to be implemented.
    def __init__(self):
        # just do it. It calls the __init__() of its parent, the nn.Module() class
        super(LinearModel,self).__init__()
        # torch.nn.Linear(1,1) creates an object named linear 
        # with weight and bias randomely initialized. 
        # In perspective of y = w*x+b: 
        # it randomly generates a learnable variable w and b. 
        # when self.linear is called, it will take the input x and do transformation of y = x*w+b
        self.linear = torch.nn.Linear(1,1)

    # when forward is called, it calls self.linear and returns the result of self.linear
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
#create an instance of LinearModel class, called model.
model = LinearModel()
# creates a criterion that measures MSE between input x and target y
# we set size_average False to make MSELoss sum the loss but not average it
# criterion = torch.nn.MSELoss(size_average=False) would return a user warning
# as size_average and reduce is in the process of being deprecated
# reduction='sum' does the same thing
criterion = torch.nn.MSELoss(reduction='sum')

# implements stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

#test model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)