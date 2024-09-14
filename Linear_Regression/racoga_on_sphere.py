import numpy as np
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *

d=1000
N=100

nb_class = 10
mean = torch.rand(nb_class,d) * 2 * d - d # random
# mean = torch.eye(nb_class,d)*d**2

#uniform spherical
# mean = torch.zeros(d)
# features_matrix,bias = sphere_uniform(d, N)
# nb_class = None


mixture_prob = np.ones(nb_class)/nb_class
# mean = (torch.diag(torch.cat((torch.ones(nb_class),torch.zeros(d-nb_class))))*500)[:nb_class,:] ### orthognal classes
features_matrix,bias = features_gaussian_mixture(d,N,mean=mean,mixture_prob=mixture_prob)

n_ech = 100
ech = sphere_uniform(d,n_ech)[0]
racoga = torch.empty(n_ech)

for i in range(n_ech):
    racoga[i] = racoga_computation(ech[i,:],N,features_matrix,bias)
    print(racoga[i])
print(racoga.min(),racoga.max(),racoga.mean())