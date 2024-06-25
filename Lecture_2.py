# Lecture on gradient descent
# we start off with randomly guessing a weight
# we update the weight as:
# w = w-alpha*(derivative of cost)/(derivative of w)
import numpy as np

# our dataset
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight w
w = 1.0

# define predicted y as x*w
def forward(x):
    return x*w

"""
# cost function
def cost(xs, ys):
    #initial cost is 0
    cost = 0
    # for all x and y in xs, ys:
    for x,y in zip(xs,ys):
        # predict y with forward()
        y_pred = forward(x)
        # the cost is squared error
        cost += (y_pred - y)**2
    # return mean squared error
    return cost / len(xs)

#gradient descent
def gradient(xs, ys):
    #initial gradient is 0
    grad = 0
    # gradient is de(cost)/de(weight)
    # cost function is sum((y_pred - y)**2)/length(x)
    # which is sum((x*w - y)**2)/length(x)
    # de(cost)/de(w) is sum(2*(x*w-y)*de(x*w-y))/length(x)
    # which is sum(2*(x*w-y)*x)/length(x)
    for x,y in zip(xs,ys):
        grad += 2*x*(x*w - y)
    return grad / len(xs)

print('Predict (before training)', 4, forward(4))
for epoch in range (100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01* grad_val
    print('Epoch: ', epoch, ', w= ', w, ', loss = ', cost_val)
print("Predict(after training):", 4, forward(4))
"""

# we don't use gradient descent that much...
# at least not this version. 
# we more often use stochastic gradient descent
# we randomly choose a loss of one sample data set 
# to calculate the gradient
# the stochastic gradient descent version of above code is as such:


# cost function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

#gradient descent
def gradient(x, y):
    return 2*x*(x*w - y)

print('Predict (before training)', 4, forward(4))

for epoch in range (100):
    for x,y in zip(x_data,y_data):
        grad = gradient(x,y)
        # update the gradient with every new set of data 
        w = w - 0.01*grad
        print("\tgrad: " , x, y, grad)
        l = loss(x,y)
    print('Progress: ', epoch, ', w= ', w, ', loss = ', l)
print("Predict(after training):", 4, forward(4))
# we could notice that the weight for each iteration is diffrent
# as it depends on different x and y. 
# that means its time complexity is relatively higher than traditional gradient descent
# so we use batch or mini batch(?) 
# we divide the whole data set into small batches (or groups) and feed them into the algorithm
# we update the weight when a whole batch is being processed, instead of one single data set in the stochastic gradient descent
# we have to manually set batch size and epoch (that makes them hyperparameters). 
# epoch determines number of complete passes through the whole training set.
# batch is the number of samples processed before model is updated.
