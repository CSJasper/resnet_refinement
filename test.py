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
from torch.utils.tensorboard import SummaryWriter

import argparse
from glob import glob
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=str, help='model name')

args = parser.parse_args()

model_name = args.n

data = pd.read_csv('val.csv', header=None)

data = data.to_numpy()

labels = data[:,0]
images = data[:,1:]

test_set = DataSet(images, labels)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=True)

model_paths = glob(os.path.join(f'{model_name}', '*.pth'))
model_paths.sort(key=lambda x: int(re.findall('_[0-9]+', x)[0][1:]))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(f'{model_name}_test')

for model_num, model_path in enumerate(model_paths):
        model = torch.load(model_path).to(device)
        
        correct_num = 0
        total_num = 0
        
        for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                
                y = torch.flatten(y)
                
                with torch.no_grad():
                        pred = model(x)
                        _, predictions = torch.max(pred, 1)
                        for label, prediction in zip(y, predictions):
                                if label.cpu().numpy() == prediction.cpu().numpy():
                                        correct_num += 1
                                total_num += 1
        acc = float(correct_num) / total_num
        print(f'acc[{model_path}] : {acc * 100:.2f}%')
        
        writer.add_scalar(f'val/acc{model_name}', acc, model_num + 1)
        
        del model
        torch.cuda.empty_cache()


writer.close()