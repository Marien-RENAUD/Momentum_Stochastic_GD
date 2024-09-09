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
from argparse import ArgumentParser
start = time.time()
parser = ArgumentParser()
parser.add_argument('--network_type', type=str, default = "CNN", choices=["CNN", "MLP"])
parser.add_argument('--batch_sample', type=str, default = "random_with_rpl", choices=["random_with_rpl", "determinist", "sort"])
parser.add_argument('--device', type=int, default = 0)
parser.add_argument('--n_epoch', type=int, default = 5)
parser.add_argument('--step', type=int, default = 100, help = "interval between each RACOGA computation")
parser.add_argument('--alg', type=str, default = "SNAG", choices = ["SNAG", "SGD", "GD", "NAG"])
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--momentum', type=float, default = 0.9)
parser.add_argument('--seed', type=int, default = 42)
hparams = parser.parse_args()

device = torch.device('cuda:'+str(hparams.device) if torch.cuda.is_available() else 'cpu')

batch_sample = hparams.batch_sample
network_type = hparams.network_type
n_epoch = hparams.n_epoch
alg = hparams.alg
momentum = hparams.momentum
lr = hparams.lr
current_seed = hparams.seed
# define network structure 

if network_type == "MLP":# MLP architecture
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
train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transform_list, download=True)

# Load results
path_results = "results/"
suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + '_seed_' + str(current_seed)
dict_path = path_results+network_type+'_n_epoch_'+str(n_epoch)+'_batch_'+batch_sample+ '_alg_' + alg  +suffix +'_dict_results.pth'
dict_results = torch.load(dict_path)
weights_trajectory = dict_results["weights_trajectory"]
loss_trajectory = dict_results["loss_trajectory"]

print("Number of iterations = {}".format(len(weights_trajectory)))

###
# Computation of RACOGA
###
post_process_loader = torch.utils.data.DataLoader(train_set, batch_size=128)

step = hparams.step # interval between each RACOGA computation
racoga_list = []
scalar_prod_list = []
iteration_list = []

if alg == "SGD" or alg == "SNAG":
    for k in tqdm(range(len(weights_trajectory)//step)):
        iteration_list.append(step*k)
        x = weights_trajectory[step*k]
        for j, param in enumerate(net.parameters()):
            param.data = x[j].to(device)
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
            #save gradients
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
        racoga_list.append(racoga.detach().cpu().numpy())
        scalar_prod_list.append(scalar_prod.detach().cpu().numpy())
else:
    for k in tqdm(range(len(weights_trajectory))):
        iteration_list.append(k)
        x = weights_trajectory[k]
        for j, param in enumerate(net.parameters()):
            param.data = x[j].to(device)
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
            #save gradients
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
        racoga_list.append(racoga.detach().cpu().numpy())
        scalar_prod_list.append(scalar_prod.detach().cpu().numpy())
duration = time.time() - start
#Save the RACOGA evolution
dict = {
    "racoga_list" : racoga_list,
    "scalar_prod_list" : scalar_prod_list,
    "iteration_list" : iteration_list,
    "computation_time" : duration
}
suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + "_seed_" + str(current_seed)
save_name = path_results+network_type+'_n_epoch_'+str(n_epoch)+'_batch_'+batch_sample+'_alg_'+ alg+ suffix
np.save(save_name+'_racoga_results.npy', dict)

# plt.plot(iteration_list, racoga_list)
# plt.xlabel("number of iterations")
# plt.ylabel("RACOGA")
# plt.savefig(save_name+"_racoga_evolution.png")

# Save info
log_print = 'racoga : '
log_print += 'datset = ' + data_choice + ', n_epoch = ' + str(n_epoch) +   ', alg = ' + alg + ', lr = ' + str(lr) + ', momentum = ' + str(momentum) + ', final training loss : ' + str(train_loss) + ', test accuracy : ' + str(test_accur) + '. Computation time : ' + str(duration)
fichier = open("log_file.txt", "a")
fichier.write(log_print)
fichier.close()