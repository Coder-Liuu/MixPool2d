import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from CustomPool2d import MedianPool2d, MaxPool2d, MixPool2d
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        mnist = np.load("./mnist.npz")
        self.x = mnist["x_train"].reshape(-1,1,28,28)
        self.x = self.x / 255.0
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(mnist["y_train"],dtype=torch.long)

    def __getitem__(self, index):
        sample = {"x": self.x[index], "y": self.y[index]}
        return sample
 
    def __len__(self):
        return len(self.x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            MixPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            MixPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32*7*7,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x) 
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
model = CNN()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
loss_func = torch.nn.CrossEntropyLoss()

for i, data in enumerate(dataloader, 0):
    inputs, labels = data["x"], data["y"]
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = loss_func(outputs, labels)
    print(f"回合{i},loss:{loss}")
    loss.backward()
    optimizer.step()
