import numpy as np
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import * ## Il y a des problèmes dans le code, les prompt ne marchent pas
## Où importer les modules?
version_bis = False # Set true to not overwrite the first data experiment
alternative_sampling = False
load_features = False
## data results folder

d,N=50,50 #choice of dimension d, number of functions N

n_sample = 10 #number of parellel occurences of stochastic algorithms
batch_size = torch.tensor([1,1,1]) #size of batch
rho = 0.1*N/batch_size
rho = torch.tensor([0.1,0.5,1])*N/batch_size
n_iter=1*10**4


if load_features:
    features = np.load("features.npy")
    features_matrix, bias = features["features_matrix"], features["bias"]
else:
    # gaussian features
    # mean = np.ones(d)*10
    mean = torch.zeros(d)*1000
    features_matrix,bias = features_gaussian(d,N,mean,generate_bias=True)
    # for j in range(1):
    #     features_matrix[j,:] /= 10**1
    # gaussian mixture features
    nb_class = 10
    # mean = torch.rand(nb_class,d) * 2 * d - d # random
    # # # nb_class = len(mean[:,0])
    # if alternative_sampling == True:
    #     features_matrix,bias = features_gaussian_mixture_det_rep(d,N,mean)  
    # else:
    #     mixture_prob = np.ones(nb_class)/nb_class
    #     # mean = (torch.diag(torch.cat((torch.ones(nb_class),torch.zeros(d-nb_class))))*500)[:nb_class,:] ### orthognal classes
    #     features_matrix,bias = features_gaussian_mixture(d,N,mean=mean,mixture_prob=mixture_prob)

    # Orthogonal features
    # features_matrix,bias = features_orthogonal(d,N,generate_lambda=True) 
    # bias = torch.zeros(N)

features = {"features_matrix" : features_matrix, "bias" : bias}
np.save("features",features)

if len(mean.shape)== 1: # 0 : unbiased, 2 : mixture
    features_type = 0
elif alternative_sampling == False:
    features_type = 1
else:
    features_type = 2

x_0 = torch.normal(torch.zeros(d),torch.ones(d))
np.save("features_type",features_type)
## We compute L and mu using AA^T or AA^T, where A is the matrix feature, depending of which matrix is the largest
## and thus the easier to compute (the largest eigenvalue is the same for both cases)
## Note that for the overparameterized case (d > N), the lowest eigenvalue of the Hessian matrix is zero, and we should instead
## consider the lower non zero eigenvalue, given by the lower eigenvalue of AA^T
if d > N:
    mu = torch.min(torch.linalg.eigh(torch.matmul(features_matrix, features_matrix.T))[0]) / N
    L = torch.max(torch.linalg.eigh(torch.matmul(features_matrix, features_matrix.T))[0]) / N
else:
    mu = torch.min(torch.linalg.eigh(torch.matmul(features_matrix.T, features_matrix))[0]) / N
    L = torch.max(torch.linalg.eigh(torch.matmul(features_matrix.T, features_matrix))[0]) / N

print("Conditionnement : ", mu/L)

vec_norm= (features_matrix**2).sum(axis=1)
# print("angle a_i : ", torch.sum(features_matrix[0,:]*features_matrix[1,:])/torch.sqrt((torch.sum(features_matrix[0,:]**2)*torch.sum(features_matrix[1,:]**2))))
arg_L_max = torch.argmax(vec_norm) 
L_max = torch.max(vec_norm)
L_sgd = N*(batch_size-1)/(batch_size*(N-1))*L + (N-batch_size)/(batch_size*(N-1))*L_max # cf. Garrigos and Gower (2024)
labels = ["GD", "Mean-SGD", "NAG"]
print("N = ",N, "cond max = ", L_max/mu)
print("NAG")
f_nag,racoga_nag = NAG(x_0,mu,L,int(n_iter*batch_size[0]/N)+1,d,N,features_matrix,bias,return_racoga = True)
print("GD")
f_gd,racoga_gd = GD(x_0,L,int(n_iter*batch_size[0]/N)+1,d,N,features_matrix,bias,return_racoga = True)
print("SGD")
print(L,L_sgd[0])
f_sgd,racoga_sgd = SGD(x_0,L_sgd[0],n_iter,n_sample,d,batch_size[0],N,features_matrix,bias,return_racoga = True,alternative_sampling=False,nb_class=nb_class)
algo = {'gd' : f_gd,'sgd' : f_sgd,'nag' : f_nag}
racoga = {'gd' : racoga_gd,'sgd' : racoga_sgd,'nag' : racoga_nag}
index = ["gd","sgd","nag"]

nb_rho = len(rho)

f_i = np.empty((n_iter,nb_rho))
print("Debut SNAG")
for i in range(nb_rho):  
    f_snag,b= SNAG(x_0,mu,L,rho[i],int(n_iter*batch_size[0]/batch_size[i])+1,n_sample,d,batch_size[i],N,features_matrix,bias,return_racoga = True,alternative_sampling=False,nb_class=nb_class)
    print(f_snag.shape)
    racoga['snag' + "batchsize=" + str(int(batch_size[i]))] = b 
    algo['snag' + "batchsize=" + str(int(batch_size[i]))] = f_snag
    labels.append("batchsize=" + str(int(batch_size[i])))
    index.append('snag' + "batchsize=" + str(int(batch_size[i])))
    print("SNAG", i)
# i=0
# rho = np.array([0.5,1])*N/batch_size
# f_snag,b = SNAG(x_0,mu,L,rho[0],n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True,alternative_sampling=False,nb_class=nb_class)
# racoga['snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k"] = b 
# algo['snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k"] = f_snag
# labels.append("rho = " + str(rho[i]*batch_size/N) + "*N/k")
# index.append('snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k")
# i=1
# f_snag,b = SNAG(x_0,mu,L,rho[0],n_iter,n_sample,d,batch_size,N,features_matrix,bias,return_racoga = True,alternative_sampling=True,nb_class=nb_class)
# racoga['snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k"] = b 
# algo['snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k"] = f_snag
# labels.append("rho = " + str(rho[i]*batch_size/N) + "*N/k")
# index.append('snag' + "rho = " + str(rho[i]*batch_size/N) + "*N/k")


root = "simul_data/batch_" 

if features_type == 0:
    suffixe = 'unbiased_'
elif features_type == 1:
    suffixe = 'gaussian_mixture_'
else:
    suffixe = 'alternative_sampling_'
nb_batch = len(batch_size)
np.save("nb_batch",nb_batch)
param = {'d' : d, 'N' : N,'n_iter' : n_iter, 'batch_size' : batch_size, 'mu' : mu, 'L' : L, 'L_max' : L_max, 'L_sgd' : L_sgd, 'rho' : rho}
torch.save(param,root + "param_"+suffixe + "batch=" + str(nb_batch) + ".pth") ### If not working, set working directory to /Linear_Regression
torch.save(algo,root +"algo_"+suffixe+ "batch=" + str(nb_batch) + ".pth")
torch.save(racoga,root +"racoga_"+ suffixe+ "batch=" + str(nb_batch) + ".pth")
torch.save(np.array(labels),root +"labels_" + "batch=" + str(nb_batch) + ".pth")
torch.save(np.array(index),root +"index_"+ "batch=" + str(nb_batch) + ".pth")

exec(open('visualization_batch.py').read()) 