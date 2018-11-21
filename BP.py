import numpy as np
import math
from math import sin, pi
import random
import matplotlib.pyplot as plt

def logsig(n):
    return np.mat([[1 / (1 + math.exp(-float(n[0][0])))], [1 / (1 + math.exp(-float(n[1][0])))]])
'''
x = np.linspace(-2, 2, 50)
y = 1 + sin(pi/4 * x)
plt.plot(x, y)
plt.show()
'''
def basic_BP(Max_iters):
    w1 = np.mat([[-0.27], [-0.41]])
    w2 = np.mat([[0.09, -0.17]])
    b1 = np.mat([[-0.48], [-0.13]])
    b2 = np.mat([[0.48]])
    rate = 0.1
    iters = 0
    while iters < Max_iters:
        prev_w1 = w1
        prev_w2 = w2
        prev_b1 = b1
        prev_b2 = b2
        p = random.uniform(-2, 2)
        prev_a1 = p
        cur_a1 = logsig(w1 * prev_a1 + b1)
        cur_a2 = w2 * cur_a1 + b2
        e = (1 + sin(pi/4 * p)) - cur_a2
#        plt.scatter(float(prev_a1), float(cur_a2), cmap = plt.cm.hot)
        cur_s = -2 * 1 * e
        prev_s = np.mat([[((1 - float(cur_a1[0][0])) * float(cur_a1[0][0])), 0], [0, (1 - float(cur_a1[1][0])) * float(cur_a1[1][0])]]) * np.transpose(w2) * cur_s
        w2 = prev_w2 - rate * cur_s * np.transpose(cur_a1)
        b2 = prev_b2 - rate * cur_s
        w1 = prev_w1 - rate * prev_s * prev_a1
        b1 = prev_b1 - rate * prev_s
        iters += 1
    return w1, b1, w2, b2

#print(basic_BP(500))

arr  = basic_BP(500)
#print(arr[0], '\n',arr[1], '\n',arr[2], '\n',arr[3],'\n')

w1 = np.mat(arr[0])
b1 = np.mat(arr[1])
w2 = np.mat(arr[2])
b2 = np.mat(arr[3])
#print(float(np.transpose(w1) * b1))

num = 0
while num < 500:
    x = random.uniform(-2, 2)
#    y = float(arr[2]) * (logsig(float(arr[0]) * x + float(arr[1]))) + float(arr[3])
#    y = arr[2] * (logsig(arr[0] * x + arr[1])) + arr[3]
    y = float(w2 * logsig(w1 * x + b1) + b2)
    plt.scatter(x, y, cmap = plt.cm.hot)
    num += 1
     
plt.show()

