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
    X,y = read_data()
    w = lassotest(X,y，alpha)
    
    return w @ data
def lassotest(X, y, alpha=0.00001, max_iter=3000, tol=1e-4):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    r = y.copy()  # residual
    Xs = np.sum(X ** 2, axis=0)
    for _ in range(max_iter):
        for j in range(n_features):
            # Update w_j
            X_j = X[:, j]
            w_j = w[j]
            w_not_j = np.delete(w, j)
            r += X_j * w_j
            w_j_new = soft_threshold(r @ X_j / n_samples, alpha / n_samples)
            r -= X_j * w_j_new
            w[j] = w_j_new
        # Check convergence
        if np.max(np.abs(w - (w - r @ X / Xs))) < tol:
            break
    return w

def soft_threshold(x, alpha):
    if x > alpha:
        return x - alpha
    elif x < -alpha:
        return x + alpha
    else:
        return 0.0
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
