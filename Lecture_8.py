# Lecture on softmax classifier
# We need something that helps us to deal with multiple outputs of a prediction
# essentially the softmax function is the percentage of output i in the total output.
# this tells us the probability of a certain output out of all possible outputs.
# keep in mind that we use exponential to make sure that the result is >=0
# how to we use it for loss function? just log it
# loss() = -y*log(y_pred)
# negative log likelihood loss(NLLLoss):
import numpy as np
y = np.array([1,0,0])
z = np.array([0.2,0.1,-0.1])
y_pred = np.exp(z)/np.exp(z).sum()
loss = (-y/np.log(z)).sum()
print(loss)

# same operation in pytorch:
# CrossEntropyLoss = LogSoftMax + NLLLoss
import torch 
y = torch.LongTensor([0])
z = torch.Tensor([[0.2,0.1,-0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z,y)
print(loss)

# in a mini-batch operation with batch size of 3
Y = torch.LongTensor([2,0,1])
Y_pred1 = torch.Tensor([[0.1,0.2,0.9],
                        [1.1,0.1,0.2],
                        [0.2,2.1,0.1]])
Y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                        [0.2,0.3,0.5],
                        [0.2,0.2,0.5]])
l1 = criterion(Y_pred1,Y)
l2 = criterion(Y_pred2,Y)
print("Batch Loss1 = ", l1.data, "\nBatch loss2 = ", l2.data)

# full example
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
# convert from PIL to Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))#0.1307 is the mean, 0.3081 is std dev
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root="../dataset/mnist/",
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)
    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
    
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target = data
        optimizer.zero_grad()

        #forward
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx % 300 ==299:
            print("[%d,%5d] loss: %3f" %(epoch+1, batch_idx+1, running_loss/300))
            running_loss=0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set: %d %%" %(100*correct/total))

if __name__ == "__main__":
    for epoch in range (10):
        train(epoch)
        test()



