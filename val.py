from audioop import avg
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import DataSet

from argparse import ArgumentParser
from glob import glob
import os
import re
from torch.utils.tensorboard import SummaryWriter

parser = ArgumentParser()

parser.add_argument('-b', type=int, default=256, help='batch_size')
parser.add_argument('--ts', type=int, default=77, help='torch manual seed')
parser.add_argument('-n', type=str, help='model name')

args = parser.parse_args()

model_name = args.n
batch_size= args.b
seed = args.ts

torch.manual_seed(seed)

data = pd.read_csv('val.csv', header=None)

data = data.to_numpy()

labels = data[:,0]
images = data[:,1:]

val_set = DataSet(images, labels)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_paths = glob(os.path.join(f'{model_name}','*.pth'))
model_paths.sort(key=lambda x: int(re.findall('_[0-9]+', x)[0][1:]))
writer = SummaryWriter(f'{model_name}_val')
criterion = nn.CrossEntropyLoss().to(device)

for model_num, model_path in enumerate(model_paths):
    model = torch.load(model_path).to(device)
    avg_loss = 0.0
    
    for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                y = torch.flatten(y)
                
                with torch.no_grad():
                    pred = model(x)
                    loss = criterion(pred, y)
                    avg_loss += loss
    
    avg_loss /= len(val_loader)
    print(f'model[{model_path}] validation loss : {avg_loss}')
    writer.add_scalar(f'loss/val{model_name}', avg_loss, model_num + 1)
    
    del model
    torch.cuda.empty_cache()
writer.close()