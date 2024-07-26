import numpy as np
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *

epsilon = 1
features_matrix = torch.tensor([[0,0.5],[epsilon,1]])
bias = torch.zeros(2)

N,d = 2,2
n_sample = 1
n_iter = 1*10**3
mu = torch.min(torch.linalg.eigh(torch.matmul(features_matrix, features_matrix.T))[0]) / N
print("mu = ",mu)
L = torch.max(torch.linalg.eigh(torch.matmul(features_matrix, features_matrix.T))[0]) / N 
print("Conditionnement : ", mu/L)
batch_size = 1
vec_norm= (features_matrix**2).sum(axis=1)
L_max = torch.max(vec_norm)
L_sgd = N*(batch_size-1)/(batch_size*(N-1))*L + (N-batch_size)/(batch_size*(N-1))*L_max # cf. Garrigos and Gower (2024)
rho = 2
nb_class = 0.5
print("L = ",L, "L_max = ",L_max)
x_0 = torch.ones(2)
x_nag,racoga_nag,f_nag= NAG(x_0,mu,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias,return_racoga = True)
x_gd,racoga_gd,f_gd = GD(x_0,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias,return_racoga = True)
x_sgd,racoga_sgd,f_sgd= SGD(x_0,L_sgd,n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True,alternative_sampling=False,nb_class=nb_class)
x_snag,b,f_snag= SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True,alternative_sampling=False,nb_class=nb_class)
print("average corelation", np.dot(features_matrix,features_matrix.T)[np.triu_indices(N,1)].mean())
print(np.where(np.dot(features_matrix,features_matrix.T)[np.triu_indices(N,1)]<0))
# n_ech = 10**3
# vec_ech = torch.empty(n_ech)
# vec_x = torch.empty((n_ech,d))
# for i in range(n_ech):
#     angle = torch.normal(torch.zeros(d))
#     x = angle/torch.sqrt(torch.sum(angle**2))
#     ### Différents choix de moyennes
#     vec_x[i,:] = x
#     vec_ech[i] = racoga_computation(x, N, features_matrix, bias)

# Hessian = torch.matmul(features_matrix.t(),features_matrix)
# print(torch.linalg.eigh(torch.matmul(features_matrix, features_matrix.T)))
# precision = 200
# grid_precision = torch.linspace(-1,1,precision)
# x_grid,y_grid = np.meshgrid(grid_precision,grid_precision)
# curvature = torch.empty((precision,precision))
# for i in range(precision):
#     for j in range(precision):
#         # gradient = grad_f(torch.tensor([x_grid[i,j],y_grid[i,j]]), N, features_matrix, bias)
#         gradient = torch.tensor([x_grid[i,j],y_grid[i,j]])
#         norm_gradient = torch.sum(gradient**2)
#         curvature[i,j] = torch.matmul(gradient.t(),torch.matmul(Hessian, gradient))/norm_gradient


plt.figure(figsize=(20,8))
plt.subplot(121)
# plt.scatter(x_grid,y_grid,c = curvature)
# print(Hessian)
# plt.set_cmap('viridis')
# plt.colorbar()
# col_vec = np.ones(n_ech)
# col_vec[np.where(vec_ech<0)] = 0
# plt.scatter(vec_x[:,0],vec_x[:,1],c=vec_ech)
# print(features_matrix)
# plt.plot(np.array([0,features_matrix[0][0]]),np.array([0,features_matrix[0][1]]))
# plt.plot(np.array([0,features_matrix[1][0]]),np.array([0,features_matrix[1][1]]))
# plt.set_cmap('gist_heat')
# plt.colorbar()
# print(torch.linspace(0,1,n_iter))
# print(x_snag[:,0].shape)

plt.scatter(x_nag[:,0],x_nag[:,1],label = "nag",marker="^",color="r")
# plt.scatter(x_gd[:,0],x_gd[:,1],label = "gd",marker="+",color="black")
plt.scatter(x_sgd[:,0],x_sgd[:,1], label = "sgd",marker="x",color="grey")
plt.scatter(x_snag[:,0],x_snag[:,1], label = "snag",marker="2",c = torch.linspace(0,1,n_iter).view(n_iter,1))
plt.colorbar()
plt.legend()
plt.subplot(122)
nb_gd_eval_det = torch.arange(0,(n_iter+1)*int(batch_size),N)
nb_gd_eval_sto = torch.arange(0,(n_iter)*batch_size,batch_size)
plt.plot(nb_gd_eval_det,torch.log(f_gd),label = "gd",color="black")
plt.plot(nb_gd_eval_sto,torch.log(f_sgd),label = "sgd",color="grey")
plt.plot(nb_gd_eval_det,torch.log(f_nag),label = "nag",color="r")
plt.plot(nb_gd_eval_sto,torch.log(f_snag),label = "snag",color="blue")
plt.legend()
plt.show()