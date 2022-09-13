import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from FullConnNet import FCN


def train(data_num):
    # 定义超参数
    batch_size = 2
    lr = 1e-4
    epoch = data_num // batch_size
    n_feature = 6
    theta_addr = '../data/train/theta.txt'
    matrix_addr = '../data/train/matrix.txt'
    theta = np.loadtxt(theta_addr, dtype=np.float).reshape(data_num, 6)
    matrix = np.loadtxt(matrix_addr, dtype=np.float).reshape(data_num, 4, 4)
    batch_theta = np.array(np.array_split(theta, epoch))
    batch_matrix = np.array(np.array_split(matrix, epoch))
    # 实例化网络
    net = FCN(n_feature)
    # 定义优化器
    optimizer = opt.Adam(net.parameters(), lr=lr)
    # 定义损失函数
    loss_func = nn.MSELoss()
    # 使用GPU加速训练
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        net.to(device)
        loss_func.to(device)
    # tqdm可以显示进度条
    for i in tqdm(range(epoch)):
        input_matrix = torch.tensor(batch_matrix[i].reshape(batch_size, 16), dtype=torch.float)
        label = torch.tensor(batch_theta[i], dtype=torch.float)
        net.train()
        # 此处填写训练代码：
        output_theta = net(input_matrix)
        loss = loss_func(output_theta, label)
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
    train(10000)
