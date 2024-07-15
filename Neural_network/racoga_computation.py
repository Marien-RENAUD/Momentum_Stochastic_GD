import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t
import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm import tqdm
from models_architecture import create_mlp, create_cnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

network_type = "CNN"
# define network structure 

if network_type == "mlp":# MLP architecture
    net = create_mlp().to(device)

if network_type == "CNN":# Light CNN architecture
    net = create_cnn().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

#Dataset load
to_tensor =  t.ToTensor()
normalize = t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
flatten =  t.Lambda(lambda x:x.view(-1))
transform_list = t.Compose([to_tensor, normalize, flatten])
train_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=True, transform=transform_list, download=True)

# Load results
path_results = "results/"
dict_results = torch.load(path_results+"dict_results.pth")
weights_trajectory = dict_results["weights_trajectory"]
loss_trajectory = dict_results["loss_trajectory"]

print("Number of iterations = {}".format(len(weights_trajectory)))

###
# Computation of RACOGA
###
post_process_loader = torch.utils.data.DataLoader(train_set, batch_size=128)

step = 100

for k in range(len(weights_trajectory)//step):
    print("Iteration {}".format(step*k))
    x = weights_trajectory[step*k]
    for j, param in enumerate(net.parameters()):
        param.data = x[j]
    sum_gradient = torch.tensor([]).to(device)
    sum_gradient_norm = 0

    for i, (batch, targets) in enumerate(post_process_loader):
        batch = batch.to(device)
        if network_type == "CNN":
            batch_size = batch.size()[0]
            batch = batch.view((batch_size, 3, 32, 32))
        output = net(batch)
        targets = targets.to(device)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        #save gardients
        gradient_i = torch.tensor([]).to(device)
        for param in net.parameters():
            gradient_i = torch.cat((gradient_i,param.grad.flatten()))
        if sum_gradient.size() != torch.Size([0]):
            sum_gradient += gradient_i
            sum_gradient_norm += torch.sum(gradient_i**2)
        else:
            sum_gradient = gradient_i

    scalar_prod = (1/2) * (torch.sum(sum_gradient**2) - sum_gradient_norm)
    racoga = scalar_prod/sum_gradient_norm
    print("SCALAR PRODUCT = {:.2f}".format(scalar_prod))
    print("RACOGA = {:.2f}".format(racoga))