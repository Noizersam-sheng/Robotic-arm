import math

import numpy as np
import pandas as pd
from tqdm import tqdm


def sin(a):
    sin_x = math.sin(a)
    return sin_x


def cos(a):
    cos_x = math.cos(a)
    return cos_x


# 位姿矩阵
def matrix_generator(theta, d, a, alpha):
    matrix = np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                       [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                       [0, sin(alpha), cos(alpha), d],
                       [0, 0, 0, 1]])
    return matrix


# 求运动学正解
def forward(theta):
    identity_matrix = np.identity(4)  # 构造单位矩阵
    a = [0, -0.42500, -0.39225, 0, 0, 0]
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.08230]
    alpha = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    for i in range(6):  # 位姿矩阵相乘
        transfer_matrix = matrix_generator(theta[i], d[i], a[i], alpha[i])
        identity_matrix = identity_matrix.dot(transfer_matrix)
    return identity_matrix


# 生成测试和训练数据
def test_generator(num):
    theta_addr = './test/theta.csv'  # 生成不同的数据记得改文件名
    matrix_addr = './test/matrix.csv'
    np.set_printoptions(precision=6)  # 控制小数点为6位
    theta = np.random.random((num, 6))
    final_position_matrix = np.empty(shape=(0, 16))
    for i in tqdm(range(num)):
        item = theta[i]
        # print(item)
        final_position = forward(item)
        # print(final_position)
        final_position = final_position.reshape(1, 16)  # 将二维矩阵拉成一维
        # print(final_position)
        final_position_matrix = np.append(final_position_matrix, final_position, axis=0)  # 将所有的一维矩阵拼接成一个二维矩阵
    # print(final_position_matrix)

    theta = pd.DataFrame(theta)
    theta.to_csv(path_or_buf=theta_addr, float_format='%6f', index=False)
    final_position_matrix = pd.DataFrame(final_position_matrix)
    final_position_matrix.to_csv(path_or_buf=matrix_addr, float_format='%6f', index=False)


if __name__ == '__main__':
    test_generator(2000)
