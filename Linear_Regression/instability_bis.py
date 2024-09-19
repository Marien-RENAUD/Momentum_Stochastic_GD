import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *
from matplotlib.lines import Line2D

d,N = 2,2
L = 10
mu = 0.1
features_matrix,bias = torch.tensor([[L,0],[0,mu]]), torch.zeros(2)


Hessian = torch.matmul(features_matrix.t(),features_matrix)
print(torch.linalg.eigh(torch.matmul(features_matrix, features_matrix.T)))
precision = 200
grid_precision = torch.linspace(-1,1,precision)
x_grid,y_grid = np.meshgrid(grid_precision,grid_precision)
curvature = torch.empty((precision,precision))
for i in range(precision):
    for j in range(precision):
        # gradient = grad_f(torch.tensor([x_grid[i,j],y_grid[i,j]]), N, features_matrix, bias)
        gradient = torch.tensor([x_grid[i,j],y_grid[i,j]])
        norm_gradient = torch.sum(gradient**2)
        curvature[i,j] = torch.matmul(gradient.t(),torch.matmul(Hessian, gradient))/norm_gradient
plt.figure(figsize = (5,5))
scatter1 = plt.scatter(x_grid, y_grid, c=curvature, cmap='viridis', label='Curvature')
cbar1 = plt.colorbar(scatter1, orientation='vertical')
cbar1.set_label('Curvature')
cbar1.set_ticks((torch.min(curvature),torch.max(curvature)))
cbar1.ax.tick_params(labelsize=number_size) 
cbar1.ax.yaxis.set_ticks_position('left')
cbar1.ax.yaxis.set_label_position('left')
cbar1.ax.set_position([0.05, 0.15, 0.02, 0.7])
plt.xlabel('x')
plt.ylabel('y')
plt.plot()