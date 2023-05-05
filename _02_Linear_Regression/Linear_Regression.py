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

    w = lassotest(X, y, 0.5)
    return w @ data


def lassotest(X, y, alpha, max_iter=5000, tol=1e-6):
    # 初始化权重向量
    w = np.zeros(X.shape[1])

    # 坐标下降算法
    for i in range(max_iter):
        w_prev = np.copy(w)
        for j in range(X.shape[1]):
            X_j = X[:, j]
            X_not_j = np.delete(X, j, axis=1)
            w_not_j = np.delete(w, j)
            c = 2 * np.dot(X_j, X_j)
            r = y - np.dot(X_not_j, w_not_j)
            z = np.dot(X_j, r)
            if z < -alpha:
                w_j = (z + alpha) / c
            elif z > alpha:
                w_j = (z - alpha) / c
            else:
                w_j = 0
            w[j] = w_j
        if np.sum(np.abs(w - w_prev)) < tol:
            break

    return w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
