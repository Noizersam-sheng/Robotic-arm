import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def length(data):
    sqr = 0
    dim = data.shape[0]
    for i in range(dim):
        sqr += data[i] ** 2
    len = sqr ** 0.5
    return len


def test(epoch):
    # 定义超参数
    loss_func = nn.MSELoss()
    theta_addr = '../data/test/theta.txt'
    matrix_addr = '../data/test/matrix.txt'
    theta = np.loadtxt(theta_addr, dtype=np.float).reshape(epoch, 6)
    matrix = np.loadtxt(matrix_addr, dtype=np.float).reshape(epoch, 4, 4)
    # 加载训练模型
    net = torch.load('./train_model/FullConnNet_25.pth')
    net.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for i in tqdm(range(epoch)):
            # 此处填写测试代码
            input_matrix = torch.tensor(matrix[i].reshape(1, 16), dtype=torch.float)
            label = torch.tensor(theta[i], dtype=torch.float)
            output_theta = net(input_matrix)
            loss = loss_func(label, output_theta)
            total_loss += loss
            len = length(label)
            total_accuracy += 1 - loss / len
        total_accuracy /= epoch
    print("loss:", loss)
    print("acc:", total_accuracy)


if __name__ == '__main__':
    # 初始化关节角度
    theta_origin = np.array([0.1, 1 / 2, 1, 1, 1, 1])
    test(2000)
