from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class DataSet(Dataset):
        def __init__(self, train_x, train_y):
                self.X = train_x
                self.Y = train_y
                self.scaler =  StandardScaler()
                
        def __getitem__(self, index: int):
                x_t = self.X[index]
                x_t = x_t.reshape(28, 28)
                x_t = self.scaler.fit_transform(x_t)  # scaling is added
                x_t = x_t.reshape(1, 28, 28)
                x_t = torch.FloatTensor(x_t)
                x_t = F.normalize(x_t, p=1, dim=1)  # normalizing term added
                y_t = torch.LongTensor([self.Y[index]])
                return x_t, y_t
        
        def __len__(self):
                return len(self.X)