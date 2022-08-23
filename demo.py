import matplotlib.pylab as plt
import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as Data
from tqdm import tqdm

from FullConnNet import FullConnNet

#########定义超参数#############
batch_size = 128
lr = 1e-3
epoch = 200
n_feature = 1

# 实例化网络
net = FullConnNet(n_feature)
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


def train_loop(loader):
    ### tqdm可以显示进度条
    for i in tqdm(range(epoch)):
        net.train()
        ### 此处填写训练代码：
        loss = None
        for step, data in enumerate(loader):
            x, y = data
            x, y = x.to(device), y.to(device)  # 将数据放入gpu中
            predict_y = net(x)
            loss = loss_func(predict_y, y)
            ### 固定三部曲：
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            # 每迭代5次打印一下损失结果
        if (i + 1) % 5 == 0:
            print('epoch: {}, loss: {:.4}'.format(i + 1, loss.data.item()))
            # 每迭代200次保存模型
            # if i % 200 == 0:
            #     torch.save(net, "./train_model/FullConnNet_{}.pth".format(epoch // 200))


def testNet(x):
    net.eval()
    with torch.no_grad():
        x = x.to(device)
        predict_y = net(x)
    return predict_y


if __name__ == '__main__':
    # 生成数据
    origin_data = np.random.uniform(-10, 10, 1000)  # 在-10和10之前产生1000个随机数
    noise = np.random.uniform(-0.1, 0.1, 1000)
    y = np.sin(origin_data) + noise  # 加入一定的噪声
    plt.scatter(origin_data, y, c='blue', label="origin")  # 原有的函数图像用蓝色表示
    # plt.show()
    origin_data = torch.from_numpy(origin_data).to(torch.float32)  # 将numpy数据转化为Tensor形式
    origin_data = origin_data.reshape(len(origin_data), 1)  # 改变数组形状为（1000，1）
    y = torch.from_numpy(y).to(torch.float32)
    y = y.reshape(len(y), 1)
    # print(y.size())
    torch_dataset = Data.TensorDataset(origin_data, y)  # 使用数据集，可以批量加载数据
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    train_loop(loader)  # 调用训练函数
    predict_y = testNet(origin_data)  # 调用测试函数
    predict_y = predict_y.cpu()  # 将预测结果转化为cpu形式，我用的是gpu训练
    plt.scatter(origin_data, predict_y, c='red', label="predict")  # 神经网络预测的图像用红褐色表示
    plt.legend()
    plt.show()

    # print(net)
    # train_loop()
