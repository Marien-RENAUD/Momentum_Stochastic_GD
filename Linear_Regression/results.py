import numpy as np
import numpy.linalg as nplinalg
import numpy.random as nprandom

from linear_regression import * ## Il y a des problèmes dans le code, les prompt ne marchent pas
## Où importer les modules?

version_bis = True # Set true to not overwrite the first data experiment
## data results folder

d,N=100,33 #choice of dimension d, number of functions N
if d>N:
    case = 0
elif d == N:
    case = 1
else:
    case = 2  # 0 : overparameterized, 1 : d=N, 2 : underparameterized

n_sample = 10 #number of parellel occurences of stochastic algorithms
batch_size = 3 #size of batch
n_iter=1*10**3

# gaussian features
mean = np.ones(d)*10
# mean = np.zeros(d)
features_matrix,bias = features_gaussian(d,N,mean)
# gaussian mixture features
mean = np.vstack([10*np.ones(d),-1*np.ones(d),-np.arange(d)])
nb_class = len(mean[:,0])
mixture_prob=np.array([0.1,0.3,0.6])
features_matrix,bias = features_gaussian_mixture(d,N,mean=mean,mixture_prob=mixture_prob)
features_matrix,bias = features_gaussian_mixture_det_rep(d,N,mean)

# Orthogonal features
# features_matrix,bias = features_orthogonal(d,N) 
if case == 2:
    bias = np.zeros(N)
if len(mean.shape)== 1 and np.any(mean != np.zeros(d)): # 0 : unbiased, 1 : biased, 2 : mixture
    features_type = 0
elif len(mean.shape)== 1 :
    features_type = 1
else: 
    features_type = 2
x_0 = nprandom.normal(0,1,d)
np.save("features_type",features_type)
np.save("case",case)
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

rho = np.array([0.5,1,1.5])*N/batch_size ## Overparameterized exemple value
if case== 2:
    rho = np.array([0.05,0.1,1])*N/batch_size ## Underparameterized exemple value

vec_norm= (features_matrix**2).sum(axis=1)
L_max = vec_norm.max()
L_sgd = N*(batch_size-1)/(batch_size*(N-1))*L + (N-batch_size)/(batch_size*(N-1))*L_max # cf. Garrigos and Gower (2024)
labels = ["GD", "Mean-SGD", "NAG"]

f_nag,racoga_nag = NAG(x_0,mu,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias,return_racoga = True)
f_gd,racoga_gd = GD(x_0,L,int(n_iter*batch_size/N)+1,d,N,features_matrix,bias,return_racoga = True)
f_sgd,racoga_sgd = SGD(x_0,L_sgd,n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True,alternative_sampling=True,nb_class=nb_class)
algo = {'gd' : f_gd,'sgd' : f_sgd,'nag' : f_nag}
racoga = {'gd' : racoga_gd,'sgd' : racoga_sgd,'nag' : racoga_nag}
index = ["gd","sgd","nag"]
for i in range(len(rho)):  
    f_snag,b = SNAG(x_0,mu,L,rho[i],n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True,alternative_sampling=True,nb_class=nb_class)
    racoga['snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k"] = b 
    algo['snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k"] = f_snag
    labels.append("rho = " + str(rho[i]*batch_size/N) + "*N/k")
    index.append('snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k")

root = "simul_data/" 

if case == 0:
    suffixe = 'overparameterized'
    if features_type == 0 :
        suffixe += '_biased_features'
    elif features_type == 1 :
        suffixe += '_unbiased_features'
    else:
        suffixe += '_gaussian_mixture'
    if version_bis == True:
        suffixe += '_bis'
elif case == 1:
    suffixe = 'd=N'
    if features_type == 0 :
        suffixe += '_biased_features'
    elif features_type == 1 :
        suffixe += '_unbiased_features'
    else:
        suffixe += '_gaussian_mixture'
    if version_bis == True:
        suffixe += '_bis'
else:
    suffixe = "underparameterized"
    if features_type == 0 :
        suffixe += '_biased_features'
    elif features_type == 1 :
        suffixe += '_unbiased_features'
    else:
        suffixe += '_gaussian_mixture'
    if version_bis == True:
        suffixe += '_bis'
print(suffixe)
param = {'d' : d, 'N' : N,'n_iter' : n_iter, 'batch_size' : batch_size, 'mu' : mu, 'L' : L, 'L_max' : L_max, 'L_sgd' : L_sgd, 'rho' : rho}
np.save(root + "param_"+suffixe,param) ### If not working, set working directory to /Linear_Regression
np.save(root +"algo_"+suffixe,algo)
np.save(root +"racoga_"+suffixe,racoga)
np.save(root +"labels",np.array(labels))
np.save(root +"index",np.array(index))

exec(open('visualization.py').read()) 