# Lecture on Dataset and DataLoader.
# test to see if github is reconfigured



import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# prepare dataset
# xy is loaded dataset from diabetes.csv. Its delimiter is ',', and its datatype is 32 digit float.
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
# every
x_data = torch.from_numpy(xy[:,:-1])
print('input data shape', x_data.shape)
y_data = torch.from_numpy(xy[:,[-1]])

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
    #two magic functions that keeps track of index and length of the dataset
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
    
dataset_d = DiabetesDataset("diabetes.csv")
# train_loader is an instance of DataLoader that loads dataset_d
# num_workers is number of threads used when reading the data.
train_loader = DataLoader(dataset = dataset_d, 
                          batch_size = 32, 
                          shuffle=True,
                          num_workers=2)

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
optimizer = torch.optim.SGD(newModel.parameters(), lr=0.01)



for epoch in range(100):
    for i, data in enumerate(train_loader,0):
        # prepare data
        inputs,labels = data
        # forward
        y_pred = newModel(inputs)
        loss = criterion(y_pred,labels)
        print(epoch, i, loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()