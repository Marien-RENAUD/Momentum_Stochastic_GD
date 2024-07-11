import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
from func import *
## Où importer les modules?

d,N=10,10 #choice of dimension d, number of functions N
n_sample = 10 #number of parellel occurences of stochastic algorithms
batch_size = 1 #size of batch
n_iter=10**3

# gaussian features
features_matrix,bias = features_gaussian(d,N)
# Orthogonal features
features_matrix,bias = features_orthogonal(d,N)
bias = np.zeros(N)

x_0 = nprandom.normal(0,1,d)
mu = nplinalg.eig(np.dot(features_matrix,features_matrix.T))[0].min()/N
L = nplinalg.eig(np.dot(features_matrix,features_matrix.T))[0].max()/N
rho = N/batch_size

vec_norm= (features_matrix**2).sum(axis=1)
L_max = vec_norm.max()
L_sgd = N*(batch_size-1)/(batch_size*(N-1))*L + (N-batch_size)/(batch_size*(N-1))*L_max # cf. Garrigos and Gower (2024)

f_nag = NAG(x_0,mu,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias)
f_gd = GD(x_0,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias)
f_sgd = SGD(x_0,L_sgd,n_iter,n_sample,d,batch_size,N,features_matrix,bias)
f_snag = SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias)

list = [f_gd,f_sgd,f_nag,f_snag]
labels = ["GD", "Mean-SGD", "NAG", "SNAG"]
plt.figure(figsize=(10,10))
color = ["black","grey","red","orange"]
nb_gd_eval = np.arange(0,n_iter+1,int(N/batch_size))
for j in range(4):
    col = color[j]
    if len(list[j].shape) >1:
        mean,min,max = np.mean(list[j],axis=1),np.min(list[j],axis=1),np.max(list[j],axis=1)
        plt.plot(np.log(mean),label=labels[j],color =col,lw=2)
        plt.plot(np.log(min),color =col,linestyle ="--")
        plt.plot(np.log(max),color =col,linestyle ="--")
    else:
        plt.plot(nb_gd_eval,np.log(list[j]),label=labels[j],color =col,lw=2)
plt.xlabel("Nb gradient evaluations",fontsize = 13)
plt.ylabel(r"$\log(f)$",fontsize= 13)
plt.legend()
plt.savefig("Linear_Regression/results/convergence_linear_regression.png")