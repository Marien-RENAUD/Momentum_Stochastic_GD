import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
from func import *
## Où importer les modules?

d,N=10,5 #choice of dimension d, number of functions N
n_sample = 10 #number of parellel occurences of stochastic algorithms
batch_size = 5 #size of batch
n_iter=10**3

features_matrix,bias = features(d,N)
vec_norm= (features_matrix**2).sum(axis=1)
L_max = vec_norm.max() ### MANQUE L_max dependant de batch size pour SGD
x_0 = np.zeros(d)
mu = nplinalg.eig(np.dot(features_matrix,features_matrix.T))[0].min()/N
L = nplinalg.eig(np.dot(features_matrix,features_matrix.T))[0].max()/N
rho = N/batch_size

f_nag = NAG(x_0,mu,L,n_iter,d,N,features_matrix,bias)
f_gd = GD(x_0,L,n_iter,d,N,features_matrix,bias)
f_sgd = SGD(x_0,L_max,n_iter,n_sample,d,batch_size,N,features_matrix,bias)
f_snag = SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias)

list = [f_gd,f_sgd,f_nag,f_snag]
labels = ["GD", "Mean-SGD", "NAG", "SNAG"]
plt.figure(figsize=(10,10))
color = ["black","grey","red","orange"]
for j in range(4):
    col = color[j]
    if len(list[j].shape) >1:
        mean,min,max = np.mean(list[j],axis=1),np.min(list[j],axis=1),np.max(list[j],axis=1)
        plt.plot(np.log(mean),label=labels[j],color =col,lw=2)
        plt.plot(np.log(min),color =col,linestyle ="--")
        plt.plot(np.log(max),color =col,linestyle ="--")
    else:
        plt.plot(np.log(list[j]),label=labels[j],color =col,lw=2)
plt.legend()
plt.show()