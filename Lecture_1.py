import numpy as np
import matplotlib.pyplot as plt

#our dataset
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    #try let x value times w
    return x*w

def loss(x, y):
    #define loss function 
    #as the error (y_pred - y) squared
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

# list of weights
w_list = []
# list of errors
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    # for all w from range (0.0, 4.1), step is 0.1:
    # print w values
    print("w is: ", w)
    # sum of loss function: this is the vallue of cost function(?)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        #for every x and y values:
        # y_pred_val is the predicted y by x*w
        y_pred_val = forward(x_val)
        #we calculate the loss value
        loss_val = loss(x_val, y_val)
        # sum up loss value
        l_sum += loss_val
        #print them out
        print('\t', x_val, y_val, y_pred_val, loss_val)
    #print mean squared error
    print("MSE=", l_sum/3)
    #add the weight to list of weights
    w_list.append(w)
    #add the cost
    mse_list.append(l_sum/3)
#plot
plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()

#personal modified to show best weight that minimizes the cost.
index_min = np.argmin(mse_list)
print("best weight is: ",w_list[index_min])