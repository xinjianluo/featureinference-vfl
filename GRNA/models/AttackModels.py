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

class Generator(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 600), 
            nn.LayerNorm(600),
            nn.ReLU(),
            
            nn.Linear(600, 200), 
            nn.LayerNorm(200),
            nn.ReLU(),
            
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            
            nn.Linear(100, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
# To be fine-tuned for different Random Forests
class FakeRandomForest(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FakeRandomForest, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.Dropout(),
            nn.ReLU(),    
            nn.Linear(200, output_dim),
            nn.Sigmoid()
        )
      
        
    def forward(self, x):
        return self.dense(x)
