import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *

d,N = 2,2
mean = torch.zeros(d)
features_matrix,bias = features_gaussian(d,N,mean)
# features_matrix[0,:] /= torch.sqrt(torch.sum(features_matrix[0,:]**2))*1.05
# features_matrix[1,:] /= torch.sqrt(torch.sum(features_matrix[1,:]**2))*1.05
bias = torch.zeros(d)

n_ech = 10**4
vec_ech = torch.empty(n_ech)
vec_x = torch.empty((n_ech,d))
for i in range(n_ech):
    angle = torch.normal(torch.zeros(d))
    x = angle/torch.sqrt(torch.sum(angle**2))*0.98
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



plt.figure(figsize=(15, 8))
number_size = 15
# Premier scatter plot avec courbure
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

# Affichage de la matrice Hessienne
print("Hessian Matrix:\n", Hessian)

# Deuxième scatter plot avec vecteurs de données
scatter2 = plt.scatter(vec_x[:, 0], vec_x[:, 1], c=vec_ech, cmap='gist_heat', label='Vector Data')
cbar2 = plt.colorbar(scatter2, orientation='vertical')
cbar2.ax.tick_params(labelsize=number_size) 
cbar2.set_ticks([torch.min(vec_ech), torch.max(vec_ech)]) 
cbar2.set_label('RACOGA')

# Ajouter des vecteurs features_matrix
plt.arrow(0,0, features_matrix[0, 0], features_matrix[0, 1],head_width=0.02, color='purple', lw=2, label='Feature 1')
plt.arrow(0,0, features_matrix[1, 0], features_matrix[1, 1],head_width=0.02, color='purple', lw=2, label='Feature 2')

# Personnalisation des axes
plt.xticks(np.linspace(-1, 1, num=3), fontsize=number_size)  # Ticks à -1, 0, 1
plt.yticks(np.linspace(-1, 1, num=3), fontsize=number_size)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Ajout de la légende et du titre
# plt.legend(loc='upper right')
# plt.title('Scatter Plot with Curvature and Vector Data')

# Affichage de la figure
plt.savefig("figure_curvature/racoga_curvature_diffsize.png")