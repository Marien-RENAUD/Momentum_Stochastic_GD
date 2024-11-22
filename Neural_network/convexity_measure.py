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
import time as time

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
parser.add_argument('--data', type=str, default = "CIFAR10", choices = ["CIFAR10", "SPHERE"])
hparams = parser.parse_args()

device = torch.device('cuda:'+str(hparams.device) if torch.cuda.is_available() else 'cpu')

batch_sample = hparams.batch_sample
network_type = hparams.network_type
n_epoch = hparams.n_epoch
alg = hparams.alg
momentum = hparams.momentum
lr = hparams.lr
current_seed = hparams.seed
data_choice = hparams.data
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

data_choice = hparams.data
if data_choice == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transform_list, download=True)
elif data_choice == "SPHERE":
    # Sphere data
    checkpoint_train = torch.load('../dataset/sphere/train_dataset_sphere.pth')
    train_set = torch.utils.data.TensorDataset(checkpoint_train['data'], checkpoint_train['labels'])

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
batch_size = 128
step = hparams.step # interval between each RACOGA computation

scalar_prod_list = []
iteration_list = []

if alg == "SGD" or alg == "SNAG":
    convexity_diff_arr = torch.empty(len(weights_trajectory)//step)
    for k in tqdm(range(len(weights_trajectory)//step)):
        iteration_list.append(step*k)
        x = weights_trajectory[step*k]
        for j, param in enumerate(net.parameters()):
            param.data = x[j].to(device)
        sum_gradient = torch.tensor([]).to(device)
        sum_loss_x = 0
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
                sum_gradient += gradient_i* batch_size
                sum_loss_x += loss.item() * batch_size
            else:
                sum_gradient = gradient_i* batch_size
                sum_loss_x = loss.item() * batch_size


        iteration_list.append(step*k)
        x_next = weights_trajectory[step*(k+1)]
        for j, param in enumerate(net.parameters()):
            param.data = x_next[j].to(device)
        sum_loss_x_next = 0

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
            
            for param in net.parameters():
                if sum_gradient.size() != torch.Size([0]):
                    sum_loss_x_next += loss.item() * batch_size
                else:
                    sum_loss_x_next = loss.item() * batch_size
            scalar_product = (sum_gradient[0]*(x[0] - x_next[0])).sum()
            for j in range(len(x)):
                scalar_product += (sum_gradient[j]*(x_next[j] - x[j])).sum()
            convexity_diff = sum_loss_x_next - (scalar_product + sum_loss_x)
            convexity_diff_arr[k] = convexity_diff#(convexity_diff.detach().cpu().numpy())
else:
    convexity_diff_arr = torch.empty(len(weights_trajectory))
    for k in tqdm(range(len(weights_trajectory))):
        iteration_list.append(k)
        x = weights_trajectory[k]
        for j, param in enumerate(net.parameters()):
            param.data = x[j].to(device)
        sum_gradient = torch.tensor([]).to(device)
        sum_gradient_norm = 0
        sum_loss_x = 0
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
                sum_gradient += gradient_i* batch_size
                sum_loss_x += loss.item() * batch_size
            else:
                sum_gradient = gradient_i* batch_size
                sum_loss_x = loss.item() * batch_size


        x_next = weights_trajectory[k]
        for j, param in enumerate(net.parameters()):
            param.data = x_next[j].to(device)
        sum_loss_x_next = 0

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
            
            for param in net.parameters():
                if sum_gradient.size() != torch.Size([0]):
                    sum_loss_x_next += loss.item() * batch_size
                else:
                    sum_loss_x_next = loss.item() * batch_size
            scalar_product = (sum_gradient[0]*(x_next[0]-x[0])).sum()
            for j in range(1,len(x)):
                scalar_product += (sum_gradient[j]*(x_next[j]-x[j])).sum()
        convexity_diff =  sum_loss_x_next - (scalar_product + sum_loss_x)
        convexity_diff_arr[k] = convexity_diff#(convexity_diff.detach().cpu().numpy())
print(convexity_diff_arr/50000)
duration = time.time() - start
#Save the RACOGA evolution
dict = {
    "convexity_diff_list" : convexity_diff_arr,
    "scalar_prod_list" : scalar_prod_list,
    "iteration_list" : iteration_list,
    "computation_time" : duration
}
suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + "_seed_" + str(current_seed)
save_name = path_results+network_type+'_n_epoch_'+str(n_epoch)+'_batch_'+batch_sample+'_alg_'+ alg+ suffix
np.save(save_name+'_convexity_test_results.npy', dict)



# Save info
log_print = '\nconvexity : '
log_print += 'datset = ' + data_choice + ', n_epoch = ' + str(n_epoch) +   ', alg = ' + alg + ', lr = ' + str(lr) + ', momentum = ' + str(momentum) +  '. Computation time : ' + str(duration)
fichier = open("log_file.txt", "a")
fichier.write(log_print)
fichier.close()