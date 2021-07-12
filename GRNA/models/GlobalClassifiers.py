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
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.models as tvmodels
import logging
from sklearn.ensemble import RandomForestClassifier


class GlobalPreModel_LR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GlobalPreModel_LR, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.dense(x)
   

   
    
class GlobalPreModel_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GlobalPreModel_NN, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 600),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(600, 300),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(300, 100),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
      
        
    def forward(self, x):
        return self.dense(x)
        
class GlobalPreModel_RF():
    def __init__(self, trees=20, depth=2, r_state=0):
        logging.critical("Creating RandomForest, with n_estimators=%d, max_depth=%d", trees, depth)
        self.rf = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=r_state)
        
      
        
     
