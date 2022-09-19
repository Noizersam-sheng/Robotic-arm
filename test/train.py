import pandas as pd
import torch.cuda
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as dt
from tqdm import tqdm
from FullConnNet import FCN
import math


def sin(a):
    sin_x = math.sin(a)
    return sin_x


def cos(a):
    cos_x = math.cos(a)
    return cos_x


# 位姿矩阵
def matrix_generator(theta, d, a, alpha):
    matrix = torch.tensor([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                           [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                           [0, sin(alpha), cos(alpha), d],
                           [0, 0, 0, 1]])
    return matrix


# 求运动学正解
def forward_train(theta):
    matrix = torch.empty((theta.shape[0], 4, 4))
    a = [0, -0.42500, -0.39225, 0, 0, 0]
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.08230]
    alpha = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    for i in range(theta.shape[0]):  # 位姿矩阵相乘
        identity_matrix = torch.eye(4)  # 构造单位矩阵
        for j in range(6):
            transfer_matrix = matrix_generator(theta[i][j], d[j], a[j], alpha[j])
            identity_matrix = torch.mm(identity_matrix, transfer_matrix)
        matrix[i] = identity_matrix
    matrix = torch.flatten(matrix, 1)
    return matrix


theta_addr = '../data/train/theta.csv'
matrix_addr = '../data/train/matrix.csv'

batch_size = 256
lr = 1e-4
n_feature = 16
epoch = 200

# 实例化网络
net = FCN(n_feature)
# 定义优化器
optimizer = opt.Adam(net.parameters(), lr=lr)
# 定义损失函数
loss_func = nn.MSELoss(reduction='sum')

# 使用GPU加速训练
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

if cuda:
    net.to(device)
    loss_func.to(device)


def train(loader):
    # tqdm可以显示进度条
    for i in tqdm(range(epoch)):
        net.train()
        # 此处填写训练代码：
        loss = None
        for step, data in enumerate(loader):
            batch_matrix, batch_theta = data  # 这里出来的数据维度已经是 （batch, n_feature)了
            batch_matrix, batch_theta = batch_matrix.to(device), batch_theta.to(device)  # 将数据放入gpu中
            batch_matrix.requires_grad = True
            predict_theta = net(batch_matrix)
            # 正向运动学
            predict_matrix = forward_train(predict_theta)
            loss = loss_func(predict_matrix, batch_matrix)
            # 固定三部曲：
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        # 每迭代50次打印一下结果
        if (i + 1) % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(i + 1, loss.data.item()))
        # 每迭代200次保存模型
        if (i + 1) % 200 == 0:
            torch.save(net, "../test/train_model/FullConnNet_{}.pth".format((i + 1) // 200))


if __name__ == '__main__':
    theta = pd.read_csv(theta_addr)
    matrix = pd.read_csv(matrix_addr)
    theta = theta.values
    matrix = matrix.values
    theta = torch.from_numpy(theta).to(torch.float32)
    matrix = torch.from_numpy(matrix).to(torch.float32)
    torch_dataset = dt.TensorDataset(matrix, theta)
    loader = dt.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # print(theta.size())
    train(loader)
