import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *
from matplotlib.lines import Line2D

def racoga_values(d,N,n_ech,features_matrix,bias):
    """
    Compute RACOGA values on the sphere
    """
    vec_ech = torch.empty(n_ech)
    vec_x = torch.empty((n_ech,d))
    for i in range(n_ech):
        angle = torch.normal(torch.zeros(d))
        x = angle/torch.sqrt(torch.sum(angle**2))*0.98
        vec_x[i,:] = x
        vec_ech[i] = racoga_computation(x, N, features_matrix, bias)
    return vec_x,vec_ech

def curvature_values(d,N,n_ech,features_matrix):
    """
    Compute curvature values along a grid
    """
    curvature = torch.empty(n_ech)
    vec_x = torch.empty((n_ech,d))
    Hessian = torch.matmul(features_matrix.t(),features_matrix)
    for i in range(n_ech):
        angle = torch.normal(torch.zeros(d))
        vec_x[i,:] = angle/torch.sqrt(torch.sum(angle**2))
        norm_x = torch.sum(vec_x[i,:]**2)
        curvature[i] = torch.matmul(vec_x[i,:].t(),torch.matmul(Hessian, vec_x[i,:]))/norm_x
    return vec_x,curvature

d,N = 2,2
mean = torch.zeros(d)
features_matrix,bias = features_gaussian(d,N,mean)
bias = torch.zeros(d)
features_matrix = torch.eye(2)*0.95
n_ech = 10**4

#-- Figures --#

plt.figure(figsize=(10, 5))
number_size = 15
label_size = 20
legend_size = 15
val_lim = 0.925
for i in range(2):
    vector_shrink = 0.85
    if i ==0:
        features_matrix = torch.eye(2)*vector_shrink
        plt.subplot(121)
    else:
        a = torch.tensor([0.2,1])
        a /=torch.sqrt(torch.sum(a**2))
        features_matrix = torch.tensor([[1,0],a])*vector_shrink
        plt.subplot(122)
    vec_ech = torch.empty(n_ech)
    vec_x = torch.empty((n_ech,d))
    vec_x,vec_ech =  racoga_values(d,N,n_ech,features_matrix,bias)
    color_map = "CMRmap"
    color_vec = "firebrick"
    scatter2 = plt.scatter(vec_x[:, 0], vec_x[:, 1], c=vec_ech, cmap=color_map,marker=".",s=200)
    cbar2 = plt.colorbar(scatter2, orientation='vertical')
    cbar2.ax.tick_params(labelsize=number_size) 
    cbar2.set_ticks([torch.min(vec_ech),0, torch.max(vec_ech)]) 
    if i == 0:
        cbar2.ax.set_visible(False)
        plt.arrow(0,0, features_matrix[0, 0], features_matrix[0, 1],head_width=0.04, color=color_vec, lw=3,label = r'Vectors $a_1,a_2$')
        plt.arrow(0,0, features_matrix[1, 0], features_matrix[1, 1],head_width=0.04, color=color_vec, lw=3)
    else:
        cbar2.set_ticklabels(["-0.1","0","0.1"])
        cbar2.set_label('RACOGA',labelpad = -10,fontsize = label_size)
        plt.arrow(0,0, features_matrix[0, 0], features_matrix[0, 1],head_width=0.04, color=color_vec, lw=3)
        plt.arrow(0,0, features_matrix[1, 0], features_matrix[1, 1],head_width=0.04, color=color_vec, lw=3)
    arrow = Line2D([0], [0], color=color_vec, marker='>', markersize=10, linestyle='None', label= r'Vectors $a_1,a_2$')

    plt.xticks(np.linspace(-1, 1, num=3), fontsize=number_size)  
    plt.yticks(np.linspace(-1, 1, num=3), fontsize=number_size)
    plt.xlim(-val_lim-0.1, val_lim+0.1)
    plt.ylim(-val_lim-0.1, val_lim+0.1)

    # Remove spines (borders)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Remove ticks
    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])
    plt.grid(True)
    if i == 0:
        plt.legend(fontsize = legend_size,loc="lower left",bbox_to_anchor=(-0.1,-0.1),frameon = False,handles = [arrow])
    else:
        plt.legend(fontsize = legend_size,loc="lower left",bbox_to_anchor=(-0.1,-0.1),frameon = False)
    plt.tight_layout()
plt.savefig("figures/curvature/racoga_only.png")
plt.figure(figsize=(10, 5))
val_lim = 1.5
for i in range(2):
    vector_shrink = 0.85
    if i ==0:
        a = torch.tensor([0.2,1])
        a /=torch.sqrt(torch.sum(a**2))
        features_matrix = torch.tensor([[1,0],a])*vector_shrink
        plt.subplot(121)
    else:
        a = torch.tensor([0.1,0.5])
        features_matrix = torch.tensor([[1,0],a])*vector_shrink
        plt.subplot(122)
    vec_ech = torch.empty(n_ech)
    vec_x = torch.empty((n_ech,d))
    vec_x,vec_ech =  racoga_values(d,N,n_ech,features_matrix,bias)
    vec_x_curvature,vec_ech_curvature = curvature_values(d,N,n_ech,features_matrix)
    vec_x_curvature *= val_lim
    color_map = "CMRmap"
    color_vec = "firebrick"
    scatter1 = plt.scatter(vec_x_curvature[:, 0], vec_x_curvature[:, 1], c=vec_ech_curvature, cmap="viridis",marker=".",s=200)
    scatter2 = plt.scatter(vec_x[:, 0], vec_x[:, 1], c=vec_ech, cmap=color_map,marker=".",s=200)
    cbar1 = plt.colorbar(scatter1, orientation='vertical')
    cbar2 = plt.colorbar(scatter2, orientation='vertical')
    cbar1.ax.tick_params(labelsize=number_size) 
    cbar1.set_ticks([torch.min(vec_ech_curvature), torch.max(vec_ech_curvature)]) 
    cbar2.ax.tick_params(labelsize=number_size) 
    cbar2.set_ticks([torch.min(vec_ech),0, torch.max(vec_ech)]) 
    if i == 0:
        cbar1.ax.set_visible(False)
        cbar2.set_label('RACOGA',fontsize = label_size)
        cbar2.ax.set_visible(False)
        plt.arrow(0,0, features_matrix[0, 0], features_matrix[0, 1],head_width=0.04, color=color_vec, lw=3,label = r'Vectors $a_1,a_2$')
        plt.arrow(0,0, features_matrix[1, 0], features_matrix[1, 1],head_width=0.04, color=color_vec, lw=3)
    else:
        cbar1.set_ticklabels([round(torch.min(vec_ech_curvature).item(),1),round(torch.max(vec_ech_curvature).item(),1)])
        cbar1.set_label('Curvature',labelpad = -10,fontsize = label_size)
        cbar2.set_ticklabels(["-0.1","0","0.1"])
        cbar2.set_label('RACOGA',labelpad = -10,fontsize = label_size)
        plt.arrow(0,0, features_matrix[0, 0], features_matrix[0, 1],head_width=0.04, color=color_vec, lw=3)
        plt.arrow(0,0, features_matrix[1, 0], features_matrix[1, 1],head_width=0.04, color=color_vec, lw=3)

    # Create custom legend handle with an arrow
    arrow = Line2D([0], [0], color=color_vec, marker='>', markersize=10, linestyle='None', label= r'Vectors $a_1,a_2$')

    plt.xticks(np.linspace(-1, 1, num=3), fontsize=number_size)  
    plt.yticks(np.linspace(-1, 1, num=3), fontsize=number_size)
    plt.xlim(-val_lim-0.1, val_lim+0.1)
    plt.ylim(-val_lim-0.1, val_lim+0.1)

    # Remove spines (borders)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Remove ticks
    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])
    plt.grid(True)
    if i == 0:
        plt.legend(fontsize = legend_size,loc="lower left",bbox_to_anchor=(-0.1,-0.1),frameon = False,handles = [arrow])
    else:
        plt.legend(fontsize = legend_size,loc="lower left",bbox_to_anchor=(-0.1,-0.1),frameon = False)
    plt.tight_layout()
plt.savefig("figures/curvature/racoga_curvature.png")