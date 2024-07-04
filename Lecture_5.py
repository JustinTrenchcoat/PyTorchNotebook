# Notes for Lecture 5: on logistic regression
# One of the important tasks in machine learning is classification.
# logistic regression does this binary classification task, as the logistic fucntion has a range of 0 to 1
# y_pred = 1/(1+exp(-a*x+b))
# logistic function essentially maps range of y = ax+b to [0,1]
# the loss function would change too. 
# we use cross-entropy to evaluate the model.
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self): 
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()
#------------------------------------------------------
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#---------------------------------------------------------------
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#---------------------------------------------------------------
x = np.linspace(0,10,200)
x_t = torch.Tensor(x).view((200,1))
print('x_t is ', x_t)
y_t = model(x_t)
y = y_t.data.numpy()
print('y is ', y)
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Prob of Passing')
plt.grid()
plt.show()