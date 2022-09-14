import pandas as pd
import torch.cuda
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as dt
from tqdm import tqdm

from FullConnNet import FCN

theta_addr = '../data/train/theta.csv'
matrix_addr = '../data/train/matrix.csv'

batch_size = 256
lr = 1e-4
epoch = 1000
n_feature = 16
epoch = 1000

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
            predict_theta = net(batch_matrix)
            # 正向运动学

            loss = loss_func(predict_theta, batch_theta)
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
