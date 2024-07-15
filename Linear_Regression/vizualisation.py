import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom
import matplotlib.pyplot as plt
from func import *
## Où importer les modules?

d,N=100 ,100 #choice of dimension d, number of functions N

n_sample = 10 #number of parellel occurences of stochastic algorithms
batch_size = 1 #size of batch
n_iter=1*10**4

# gaussian features
features_matrix,bias = features_gaussian(d,N)
# Orthogonal features
features_matrix,bias = features_orthogonal(d,N)
# lambda_vec = np.ones(N)
# features_matrix = np.eye(N,d)*lambda_vec.reshape(N,1)
bias = np.zeros(N)
x_0 = nprandom.normal(0,1,d)

## We compute L and mu using AA^T or AA^T, where A is the matrix feature, depending of which matrix is the largest
## and thus the easier to compute (the largest eigenvalue is the same for both cases)
## Note that for the overparameterized case (d > N), the lowest eigenvalue of the Hessian matrix is zero, and we should instead
## consider the lower non zero eigenvalue, given by the lower eigenvalue of AA^T
if d>N:
    mu = nplinalg.eig(np.dot(features_matrix,features_matrix.T))[0].min()/N
    L = nplinalg.eig(np.dot(features_matrix,features_matrix.T))[0].max()/N
else: 
    mu = nplinalg.eig(np.dot(features_matrix.T,features_matrix))[0].min()/N
    L = nplinalg.eig(np.dot(features_matrix.T,features_matrix))[0].max()/N
print("Conditionnement : ", mu/L)

rho = np.array([0.8,1,1.2,1.5])*N/batch_size
vec_norm= (features_matrix**2).sum(axis=1)
L_max = vec_norm.max()
L_sgd = N*(batch_size-1)/(batch_size*(N-1))*L + (N-batch_size)/(batch_size*(N-1))*L_max # cf. Garrigos and Gower (2024)
labels = ["GD", "Mean-SGD", "NAG"]

f_nag,racoga_nag = NAG(x_0,mu,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias,return_racoga = True)
f_gd,racoga_gd = GD(x_0,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias,return_racoga = True)
f_sgd,racoga_sgd = SGD(x_0,L_sgd,n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True)
list = [f_gd,f_sgd,f_nag]
racoga_snag = []
for i in range(len(rho)):  
    f_snag,b = SNAG(x_0,mu,L,rho[i],n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True)
    racoga_snag.append(b) 
    list.append(f_snag)
    labels.append("rho = " + str(rho[i]*batch_size/N) + "*N/k")
plt.figure(figsize=(20,10))
nb_alg = len(list) 
nb_rho = len(rho)
nb_gd_eval = np.arange(0,n_iter+1,int(N/batch_size))
k = 0
for j in range(nb_alg):
    if j==0:
        col = "black"
    elif j ==1:
        col = "grey"
    elif j == 2:
        col = "red"
    else:
        col = (0.25, k/nb_rho ,1-k/nb_rho)
        k+=1
    if len(list[j].shape) >1:
        mean_alg,min_alg,max_alg = np.mean(list[j],axis=1),np.min(list[j],axis=1),np.max(list[j],axis=1)
        plt.plot(np.log(mean_alg),label=labels[j],color =col,lw=2)
        plt.plot(np.log(min_alg),color =col,linestyle ="--")
        plt.plot(np.log(max_alg),color =col,linestyle ="--")
    else:
        print("a")
        # plt.plot(nb_gd_eval,np.log(list[j]),label=labels[j],color =col,lw=2)
plt.xlabel("Nb gradient evaluations",fontsize = 13)
plt.ylabel(r"$\log(f)$",fontsize= 13)
plt.legend()
plt.title("Algorithms convergence, N = " + str(N) + ", d = " + str(d))
plt.savefig("Linear_Regression/results/convergence_linear_regression_overparameterized.png")
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.yticks((racoga_gd.min(),0,racoga_gd.max()))
plt.plot(racoga_gd,label = "GD",color="b")
plt.plot(racoga_nag,label = "NAG",color="r")
plt.title("RACOGA condition along iterations",fontsize = 10)
# print(np.where(racoga_snag<0))
plt.legend()
plt.subplot(122)
for j in range(len(rho)):
    col = (0.5, j/nb_alg ,1-j/nb_alg)
    plt.hist(racoga_snag[j],bins=np.linspace(racoga_snag[j].min(),racoga_snag[j].max(),50),edgecolor="black",facecolor = col,label = labels[j+3])
plt.title("RACOGA condition number along iterations of SNAG",fontsize=10)
# plt.plot(racoga_sgd)
# plt.plot(racoga_snag)
# print("racoga sgd", racoga_sgd.min())
# print("racoga snag", racoga_snag.min())
plt.legend()
plt.savefig("Linear_Regression/results/racoga_overparameterized.png")
print("L_max = ", L_max," L = ", L, "rhoL = ", rho*L)
print(2*L_max < rho*L)