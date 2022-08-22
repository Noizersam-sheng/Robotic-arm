import torch.nn as nn


class FullConnNet(nn.Module):
    def __init__(self, n_feature):
        super(FullConnNet, self).__init__()
        self.h1 = nn.Sequential(nn.Linear(n_feature, 16), nn.BatchNorm1d(16), nn.ReLU(True)) # Sequential相当于将网络层顺序拼接
        self.h2 = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(True))
        self.h3 = nn.Sequential(nn.Linear(32, 64), nn.BatchNorm1d(64), nn.Sigmoid())
        self.h4 = nn.Linear(64, n_feature)

    def forward(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        return x
