import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm


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
    theta_addr = '../data/test/theta.csv'
    matrix_addr = '../data/test/matrix.csv'
    theta = np.array(pd.read_csv(theta_addr)).reshape(epoch, 6)
    matrix = np.array(pd.read_csv(matrix_addr)).reshape(epoch, 4, 4)
    # 加载训练模型
    net = torch.load('./train_model/FullConnNet_1.pth')
    net.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for i in tqdm(range(epoch)):
            # 此处填写测试代码
            input_matrix = torch.tensor(matrix[i].reshape(1, 16), dtype=torch.float).to('cpu')
            label = torch.tensor(theta[i], dtype=torch.float).to('cpu')
            output_theta = net(input_matrix)
            output_matrix = forward_train(output_theta)
            loss = loss_func(input_matrix, output_matrix)
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
