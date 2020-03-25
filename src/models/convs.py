import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes: int = 1):
        super(CNN, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=4,padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=2,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.linear = nn.Linear(1 * 128 * 3 * 3, n_classes)
        
    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)
        out = self.seq3(out)
        out = self.seq4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
