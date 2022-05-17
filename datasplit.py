import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


path = 'data/emnist-balanced-train.csv'
data = pd.read_csv(path, header=None, dtype=np.int64)

labels = data.values[:,0]
images = data.values[:,1:]

train_test_shuffle_seed = 77
train_val_shuffle_seed = 11
test_data_ratio = 0.15
val_data_ratio = 0.2

train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=test_data_ratio, random_state=train_test_shuffle_seed)
train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=val_data_ratio, random_state=train_val_shuffle_seed)

with open('./_train.csv', 'w') as f:
        for i, label in enumerate(train_y):
                line = str(label) + ','
                for val in train_x[i]:
                        line += str(val) + ','
                line = line[:-1]
                
                line += '\n'
                
                f.write(line)

with open('./_val.csv', 'w') as f:
        for i, label in enumerate(val_y):
                line = str(label) + ','
                for val in val_x[i]:
                        line += str(val) + ','
                line = line[:-1]
                
                line += '\n'
                
                f.write(line)

with open('./_test.csv', 'w') as f:
        for i, label in enumerate(test_y):
                line = str(label) + ','
                for test in test_x[i]:
                        line += str(test) + ','
                line = line[:-1]
                
                line += '\n'
                
                f.write(line)
