# lecture on back propagation
# example of two layer NN:
# y_pred = W_2*(W_1*x + b_1) + b_2
# we observe that this combination of linear transformation could be intergrated into one.
# to make the network more complex as to mimic the neural network, we apply some non-linear transformation to the output of each layer of network
# such as sigmoid(W_2*sigmoid(W_1*x + b_1)+b_2)
# we utilize chain rule to get derivatives of each layer of network.
# Tensor: important component in PyTorch in constructing dynamic computational graph.
# It contains data and grad, which stores the value of node and gradient with respect to loss 
# here is the implementation of back propagration:

import torch

# our dataset
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#set initial guess of w
w = torch.Tensor([1.0])
#indicates that w needs to calculate the gradient
w.requires_grad = True

def forward(x):
    #x is now a tensor as well. 
    return x*w
# when operating on tensors, PyTorch would create a computation graph which takes up space in memory
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

print("predict (before training)", 4, forward(4).item())
for epoch in range (100):
    for x,y, in zip(x_data,y_data):
        l = loss(x,y)
        # once backward() is executed, the computation graph is discarded and new one is ready to be contructed.
        # so that it does not take up memory space.
        l.backward()
        print("\tgrad: ", x,y,w.grad.item())
        # update weight value with our calculated gradient.
        w.data = w.data - 0.01*w.grad.data

        #set gradient back to zero for next computation.
        w.grad.data.zero()

    print('progress: ', epoch, l.item())

print("predict (after training)", 4, forward(4).item())