import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, n_feature):
        super(FCN, self).__init__()
        self.h1 = nn.Sequential(nn.Linear(16, 256), nn.BatchNorm1d(256), nn.ReLU(True))  # Sequential相当于将网络层顺序拼接
        self.h2 = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(True))
        self.h3 = nn.Sequential(nn.Linear(64, 16), nn.BatchNorm1d(16), nn.ReLU(True))
        self.h4 = nn.Linear(16, 6)

    def forward(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        return x
