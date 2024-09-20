import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *
from matplotlib.lines import Line2D
import matplotlib as mpl

d,N = 2,2
lambda_max = 10.
lamba_min = 3.
epsilon = 5
# features_matrix,bias = torch.tensor([[lambda_max,0],[0,lamba_min]]), torch.zeros(2)
features_matrix,bias = torch.tensor([[lambda_max,epsilon],[0,lamba_min]]), torch.zeros(2)
Hessian = torch.matmul(features_matrix.t(),features_matrix)
# L = lambda_max**2/2
# mu = lamba_min**2/2
L = nplinalg.eig(Hessian)[0].max()/2
mu = nplinalg.eig(Hessian)[0].min()/2


#-- Computation of curvature around the solution --#


precision = 200
grid_precision = torch.linspace(-1,1,precision)
x_grid,y_grid = np.meshgrid(grid_precision,grid_precision)
curvature = torch.empty((precision,precision))
for i in range(precision):
    for j in range(precision):
        gradient = torch.tensor([x_grid[i,j],y_grid[i,j]])
        norm_gradient = torch.sum(gradient**2)
        curvature[i,j] = torch.matmul(gradient.t(),torch.matmul(Hessian, gradient))/norm_gradient

#-- Computation of first iterations of GD, SGD, NAG and SNAG --#

shrink_factor = 0.8
x_0 = torch.ones(2)*shrink_factor
x_0 = torch.tensor([0.,1.])*shrink_factor
rho = 2
L_max = lambda_max**2 + epsilon**2
n_sample = 1 #number of parellel occurences of stochastic algorithms
batch_size = 1 #size of batch
n_iter=2*10**1
nb_class = None
f_nag,x_nag = NAG(x_0,mu,L,n_iter,d,N,features_matrix,bias,return_traj=True)
f_gd,x_gd = GD(x_0,L,n_iter,d,N,features_matrix,bias,return_traj=True)
f_sgd,x_sgd = SGD(x_0,L_max,n_iter,n_sample,d,batch_size,N,features_matrix,bias,alternative_sampling=False,nb_class=nb_class,return_traj=True)
f_snag,x_snag = SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias,alternative_sampling=False,nb_class=nb_class,L_max=L_max,return_traj=True)
print(f_snag[-1])
#-- Figures --#

number_size = 15
label_size = 20
legend_size = 15
label_pad = -20
plt.figure(figsize = (10,5))
scatter1 = plt.scatter(x_grid, y_grid, c=curvature, cmap='pink_r')
cbar1 = plt.colorbar(scatter1, orientation='vertical')
cbar1.set_label('Curvature',fontsize = label_size,labelpad = label_pad)
cbar1.set_ticks((torch.min(curvature),torch.max(curvature)))
cbar1.ax.tick_params(labelsize=number_size) 
cbar1.ax.set_position([0.05, 0.15, 0.02, 0.7])
plt.plot(x_snag[:,0],x_snag[:,1],label = "SNAG",color = "blue",marker = "x")
plt.plot(x_nag[:,0],x_nag[:,1],label = "NAG",color = "r",marker='+')
plt.plot(x_gd[:,0],x_gd[:,1],label = "GD",color = "purple",marker = "+")
plt.text(shrink_factor, shrink_factor+0.05, r'$x_0$', fontsize=12, color='black')
for spine in plt.gca().spines.values():
        spine.set_visible(False)
plt.ylim(-0.1, 1)
plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])
plt.subplots_adjust()
print(mpl.rcParams['xtick.direction'])
plt.legend(fontsize = legend_size)
plt.savefig("figures/instability/curvature.png")