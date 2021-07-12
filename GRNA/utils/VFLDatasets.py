import configparser
import logging

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import torch.autograd as autograd
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.models as tvmodels
from datetime import datetime

class ExperimentDataset(Dataset):

    def __init__(self, datafilepath):
        full_data_table = np.genfromtxt(datafilepath, delimiter=',')
        data = torch.from_numpy(full_data_table).float()
        self.samples = data[:, :-1]
        # permuate columns 
        batch, columns = self.samples.size()
        permu_cols = torch.randperm(columns)
        logging.critical("Dataset column permutation is: \n %s", permu_cols)
        self.samples = self.samples[:, permu_cols]
        
        self.labels = data[:, -1]
        min, _ = self.samples.min(dim=0)
        max, _ = self.samples.max(dim=0)
        self.feature_min = min
        self.feature_max = max

        self.samples = (self.samples - self.feature_min)/(self.feature_max-self.feature_min)
        logging.critical("Creating dataset, len(samples): %d; positive labels sum: %d", len(self.labels), (self.labels > 0).sum().item())
        self.mean_attr = self.samples.mean(dim=0)
        self.var_attr = self.samples.var(dim=0)
                      
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
    
class FakeDataset(Dataset):

    def __init__(self, length, x_dim, RF):
        self.x = torch.rand(length, x_dim)
        y = RF.predict_proba(self.x.numpy())
        self.y = torch.tensor(y).float() 
        logging.critical("Creating FakeDataset (%s, %s)", self.x.shape, self.y.shape)
                      
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]