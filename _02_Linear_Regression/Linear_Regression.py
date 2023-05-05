# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y = read_data()
    l = -0.1
    weight = np.dot(np.linalg.inv((np.dot(x.T,x)+np.dot(l,np.eye(6)))),np.dot(x.T,y))
    return weight @ data
    
def lasso(data):
    x, Y = read_data()
    weight = data
    y = np.dot(weight, x.T)
    l = 3000
    rate = 0.00000000001
    for i in range(int(2e5)):
        y = np.dot(weight, x.T)
        dw = np.dot(y - Y, x) + l * np.sign(weight)
        weight = weight * (1 - (rate * l / 6)) - dw * rate
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
