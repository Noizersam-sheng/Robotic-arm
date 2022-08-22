import torch.cuda
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

from FullConnNet import FullConnNet

#########定义超参数#############
batch_size = 16
lr = 1e-4
epoch = 1000

net = FullConnNet(4)
optimizer = opt.Adam(net.parameters(), lr=lr)
loss_func = nn.MSELoss()

### 使用GPU加速训练
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

if cuda:
    net.to(device)
    loss_func.to(device)


def train_loop():
    ### tqdm可以显示进度条
    for i in tqdm(range(epoch + 1)):
        ### 此处填写训练代码：
        net.train()

        loss = loss_func()
        ### 固定三部曲：
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        if i % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss_func.data.item()))

        if i % 200 == 0:
            torch.save(net, "./train_model/FullConnNet_{}.pth".format(epoch // 200))


if __name__ == '__main__':
    print(net)
    train_loop()
