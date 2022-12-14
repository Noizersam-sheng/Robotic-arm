import torch.cuda
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

from FullConnNet import FullConnNet

#########定义超参数#############
batch_size = 16
lr = 1e-4
epoch = 1000
n_feature = 4

# 实例化网络
net = FullConnNet(n_feature)
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


def train_loop():
    ### tqdm可以显示进度条
    for i in tqdm(range(epoch)):
        net.train()
        ### 此处填写训练代码：

        loss = loss_func()
        ### 固定三部曲：
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        # 每迭代50次打印一下结果
        if (i+1) % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(i+1, loss.data.item()))
        # 每迭代200次保存模型
        if (i+1) % 200 == 0:
            torch.save(net, "./train_model/FullConnNet_{}.pth".format((i+1) // 200))


if __name__ == '__main__':
    print(net)
    train_loop()
