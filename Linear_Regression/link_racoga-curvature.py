import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *

d,N = 2,2
mean = torch.zeros(d)
features_matrix,bias = features_gaussian(d,N,mean)
# features_matrix[0,:] /= torch.sqrt(torch.sum(features_matrix[0,:]**2))
# features_matrix[1,:] /= torch.sqrt(torch.sum(features_matrix[1,:]**2))
bias = torch.zeros(d)

n_ech = 10**3
vec_ech = torch.empty(n_ech)
vec_x = torch.empty((n_ech,d))
for i in range(n_ech):
    angle = torch.normal(torch.zeros(d))
    x = angle/torch.sqrt(torch.sum(angle**2))
    ### Différents choix de moyennes
    vec_x[i,:] = x
    vec_ech[i] = racoga_computation(x, N, features_matrix, bias)

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


plt.figure(figsize=(15,8))
plt.scatter(x_grid,y_grid,c = curvature)
print(Hessian)
plt.set_cmap('viridis')
plt.colorbar()
col_vec = np.ones(n_ech)
col_vec[np.where(vec_ech<0)] = 0
plt.scatter(vec_x[:,0],vec_x[:,1],c=vec_ech)
print(features_matrix)
plt.plot(np.array([0,features_matrix[0][0]]),np.array([0,features_matrix[0][1]]))
plt.plot(np.array([0,features_matrix[1][0]]),np.array([0,features_matrix[1][1]]))
plt.set_cmap('gist_heat')
plt.colorbar()
plt.show()