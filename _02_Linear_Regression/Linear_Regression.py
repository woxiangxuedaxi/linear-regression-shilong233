# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y = read_data()
    
    # 将 X 展平成二维数组
    n_samples = X.shape[0]
    X_flat = X.reshape((n_samples, -1))

    # 设置岭回归的超参数 alpha
    alpha = -0.1

    # 计算 X_flat 的转置矩阵
    X_flat_transpose = np.transpose(X_flat)

    # 计算矩阵 X_flat_transpose * X_flat + alpha * I 的逆矩阵
    inverse = np.linalg.inv(X_flat_transpose.dot(X_flat) + alpha * np.identity(X_flat.shape[1]))

    # 计算最优权重
    w = inverse.dot(X_flat_transpose).dot(y)
    return w @ data
   
def lasso(data):
    X, y = read_data()

    # 设置超参数
    alpha = 0.1
    learning_rate = 0.01
    n_iterations = 1000

    # 初始化权重
    np.random.seed(0)
    w = np.random.randn(X_flat.shape[1])

    # 实现 Lasso 回归的梯度下降算法
    for i in range(n_iterations):
        # 计算预测值
        y_pred = X_flat.dot(w)
    
        # 计算误差
        error = y - y_pred
    
        # 计算 L1 正则化项的梯度
        l1_grad = alpha * np.sign(w)
    
        # 计算权重的梯度
        grad = -2 * X_flat.T.dot(error) / n_samples + l1_grad
    
        # 更新权重
        w -= learning_rate * grad
    return w @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
