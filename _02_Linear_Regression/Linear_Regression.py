# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X, y = read_data()

    w = lassotest(X, y, 0.9)
    return w @ data


def lassotest(X, y, alpha, max_iter=5000, tol=1e-19):
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
