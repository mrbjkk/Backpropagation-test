from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

#fig = plt.figure()
#ax = Axes3D(fig)

def func(dataset):
    return np.mat([[10, -6],[-6, 10]]) * np.mat([[float(dataset[0][0])],[float(dataset[1][0])]]) + np.mat([[4],[4]])

x = np.arange(-2.5, 0, 0.01)
y = np.arange(-2.5, 0, 0.01)
X, Y = np.meshgrid(x,y)
Z = 5* X ** 2 - 6 * X * Y + 5 * Y ** 2 + 4 * X + 4 * Y
#plt.contourf(X, Y, Z, zdir = 'z', offset = -4, cmap=plt.cm.hot)
plt.contour(X, Y, Z, colors = 'black')
#ax.plot_surface(X, Y, Z,rstride = 1, cstride = 1,cmap = plt.get_cmap('rainbow'))
#plt.show()

def grad_d(cur_x, rate, max_iters):
    iters = 0
    while iters < max_iters:
        prev_x = cur_x
        cur_x = prev_x - rate * func(prev_x)
        iters = iters + 1
#        print(cur_x)
        plt.scatter(float(cur_x[0][0]), float(cur_x[1][0]), c = 'black')
    return cur_x

print(grad_d(np.mat([[-1],[-2.5]]), rate = 0.01, max_iters = 200))

plt.show()
