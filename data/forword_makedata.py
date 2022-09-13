import math
import random
import numpy as np


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
def forward(theta, num):
    matrix = np.zeros(shape=(num, 4, 4))
    a = [0, -0.42500, -0.39225, 0, 0, 0]
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.08230]
    alpha = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    for i in range(theta.shape[0]):
        item = theta[i]
        t_01 = matrix_generator(item[0], d[0], a[0], alpha[0])
        t_12 = matrix_generator(item[1], d[1], a[1], alpha[1])
        t_23 = matrix_generator(item[2], d[2], a[2], alpha[2])
        t_34 = matrix_generator(item[3], d[3], a[3], alpha[3])
        t_45 = matrix_generator(item[4], d[4], a[4], alpha[4])
        t_56 = matrix_generator(item[5], d[5], a[5], alpha[5])
        matrix[i] = t_01.dot(t_12.dot(t_23).dot(t_34).dot(t_45).dot(t_56))
    return matrix


# 生成训练数据
def train_generator(train_addr, num):
    theta_addr = train_addr + 'theta.txt'
    matrix_addr = train_addr + 'matrix.txt'
    theta = np.random.random((num, 6))
    matrix = forward(theta, num)
    np.savetxt(theta_addr, theta, fmt='%6f')
    with open(matrix_addr, 'w') as file:
        for item in matrix:
            np.savetxt(file, item, fmt='%6f')


# 生成测试数据
def test_generator(test_addr, num):
    theta_addr = test_addr + 'theta.txt'
    matrix_addr = test_addr + 'matrix.txt'
    theta = np.random.random((num, 6))
    matrix = forward(theta, num)
    np.savetxt(theta_addr, theta, fmt='%6f')
    with open(matrix_addr, 'w') as file:        for item in matrix:
            np.savetxt(file, item, fmt='%6f')


if __name__ == '__main__':
    train_addr = './train/'
    test_addr = './test/'
    train_generator(train_addr, 10000)
    test_generator(test_addr, 2000)
