import numpy as np
import numpy.random as nprandom
import matplotlib.pyplot as plt
import torch 
from linear_regression import *
from argparse import ArgumentParser

# Define Parser arguments
parser = ArgumentParser()
parser.add_argument('--data_type', type=str, default = "both", choices=["both","gaussian_mixture", "sphere"])
parser.add_argument('--n_ech', type=int, default = 100)
hparams = parser.parse_args()
data_type = hparams.data_type
n_ech = hparams.n_ech
d=1000
N=100
label_size = 20
legend_size = 15
number_size = 15
labelpad_x = -10
if data_type == "both":

    mean = torch.zeros(d)
    features_matrix,bias = sphere_uniform(d, N)
    bias = torch.zeros(N)
    nb_class = None
    ech = sphere_uniform(d,n_ech)[0]
    racoga = torch.empty(n_ech)
    for i in range(n_ech):
        racoga[i] = racoga_computation(ech[i,:],N,features_matrix,bias)
    
    plt.figure(figsize=(5,5))
    col = "deepskyblue"
    plt.hist(racoga,bins = "sqrt",edgecolor=None,facecolor = col,density = True,alpha = 1)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    plt.xticks((-0.05,0,0.15),["-0.05", "0", "0.15"], fontsize = number_size)
    plt.yticks((0,20), fontsize = number_size)
    plt.savefig("figures/curvature/racoga_sphere.png")
    plt.figure(figsize=(5,5))
    nb_class = 10
    mean = torch.rand(nb_class,d) * 2 * d - d # random
    mixture_prob = np.ones(nb_class)/nb_class
    features_matrix,bias = features_gaussian_mixture(d,N,mean=mean,mixture_prob=mixture_prob)
    racoga = torch.empty(n_ech)
    for i in range(n_ech):
        racoga[i] = racoga_computation(ech[i,:],N,features_matrix,bias)

    plt.hist(racoga,bins = "sqrt",edgecolor=None,facecolor = col,density = True,alpha = 1)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    plt.xticks((0,4),["0", "4"], fontsize = number_size)
    plt.yticks((0,1), fontsize = number_size)
    plt.legend(fontsize = legend_size,frameon = False)
    plt.savefig("figures/curvature/racoga_sphere_mixture.png")
elif data_type == "sphere":
    mean = torch.zeros(d)
    features_matrix,bias = sphere_uniform(d, N)
    nb_class = None
    ech = sphere_uniform(d,n_ech)[0]
    racoga = torch.empty(n_ech)
    for i in range(n_ech):
        racoga[i] = racoga_computation(ech[i,:],N,features_matrix,bias)
    plt.figure(figsize=(5,5))
    plt.hist(racoga,bins = "sqrt")
    plt.savefig("figures/curvature/racoga_sphere.png")
else:
    nb_class = 10
    mean = torch.rand(nb_class,d) * 2 * d - d # random
    mixture_prob = np.ones(nb_class)/nb_class
    features_matrix,bias = features_gaussian_mixture(d,N,mean=mean,mixture_prob=mixture_prob)
    ech = sphere_uniform(d,n_ech)[0]
    racoga = torch.empty(n_ech)
    for i in range(n_ech):
        racoga[i] = racoga_computation(ech[i,:],N,features_matrix,bias)
    plt.figure(figsize=(5,5))    
    plt.hist(racoga,bins = "sqrt")
    plt.savefig("figures/curvature/racoga_sphere_mixture.png")