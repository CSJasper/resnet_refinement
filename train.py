import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import DataSet
import argparse
from arch import custom_resnet18
from arch import custom_resnet34
from arch import custom_resnet50
from arch import simple

from torch.utils.tensorboard import SummaryWriter

import os

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, help='--lr : learning rate')
parser.add_argument('-e', type=int, help='-e : epochs')
parser.add_argument('-b', type=int, help='-b : batch size')
parser.add_argument('--ts', type=int, default=77, help='torch manual seed')
parser.add_argument('-n', type=str, help='model name')
parser.add_argument('--itv', type=int, default=0, help='intervention epochs (train from this epoch)')

args = parser.parse_args()
model_name = args.n
inter_epoch = args.itv

torch.manual_seed(args.ts)

data = pd.read_csv('train.csv', header=None)

data = data.to_numpy()

labels = data[:,0]
images = data[:,1:]

train_set = DataSet(images, labels)
train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(model_name)

if not os.path.exists(model_name):
        os.makedirs(model_name)   

model = custom_resnet50().to(device) if inter_epoch == 0 else torch.load(f'{model_name}/trained_{inter_epoch}.pth')
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(inter_epoch, args.e):
        avg_loss = 0.0
        for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                
                y = torch.flatten(y)
                
                optimizer.zero_grad()
                
                pred = model(x)
                
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                avg_loss += loss
        
        avg_loss /= len(train_loader)
        
        writer.add_scalar(f'loss/train{model_name}', avg_loss, epoch + 1)
        torch.save(model, f'./{model_name}/trained_{epoch + 1}.pth')
        
        print(f'Epoch[{epoch + 1} / {args.e}] train avg loss {model_name} : {avg_loss}')

writer.close()