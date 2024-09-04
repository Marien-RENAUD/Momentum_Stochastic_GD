import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t
import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm.auto import tqdm
from models_architecture import create_mlp, create_cnn
from argparse import ArgumentParser
from utils import sort_batches

# Define Parser arguments
parser = ArgumentParser()
parser.add_argument('--network_type', type=str, default = "CNN", choices=["CNN", "MLP"])
parser.add_argument('--batch_sample', type=str, default = "random_with_rpl", choices=["random_with_rpl", "determinist", "sort", "random_sort"])
parser.add_argument('--device', type=int, default = 0)
parser.add_argument('--n_epoch', type=int, default = 5)
# parser.add_argument('--alg', type=str, default = "SNAG", choices = ["SNAG", "SGD", "GD", "NAG"])
hparams = parser.parse_args()

device = torch.device('cuda:'+str(hparams.device) if torch.cuda.is_available() else 'cpu')
# define network structure 
network_type = hparams.network_type

if network_type == "MLP":# MLP architecture
    net = create_mlp().to(device)

if network_type == "CNN":# Light CNN architecture
    net = create_cnn().to(device)

criterion = nn.CrossEntropyLoss()

momentum = None
alg = hparams.alg

if alg == "SGD" or alg == "SNAG":
    lr = 0.01
if alg == "GD":
    lr = 2
if alg == "NAG":
    lr = 1


if alg == "SGD" or alg == "GD":
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)

if alg == "SNAG" or alg == "NAG":
    momentum = 0.9
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum, nesterov = "True")

###
# DATA PRE-PROCESSING
###

# load data
to_tensor =  t.ToTensor()
normalize = t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
flatten =  t.Lambda(lambda x:x.view(-1))

transform_list = t.Compose([to_tensor, normalize, flatten])
train_set = torchvision.datasets.CIFAR10(root='/beegfs/jhermant/Momentum_Stochastic_GD/dataset', train=True, transform=transform_list, download=True)
test_set = torchvision.datasets.CIFAR10(root='/beegfs/jhermant/Momentum_Stochastic_GD/dataset', train=False, transform=transform_list, download=True)


# Load tensors from the file
checkpoint = torch.load('/beegfs/jhermant/Momentum_Stochastic_GD/dataset/sphere/dataset_sphere.pth')
# Recreate the tensordataset
loaded_dataset = torch.utils.data.TensorDataset(checkpoint['data'], checkpoint['labels'])
train_size = int(5/6 * len(loaded_dataset))
test_size = len(loaded_dataset) - train_size
# Split dataset between training set and test set
train_dataset, test_dataset = torch.utils.data.random_split(loaded_dataset, [train_size, test_size])







batch_size_train = 64
if alg == "GD" or alg == "NAG":
    batch_size_train = 5000
batch_size_test = 64
## Sphere data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train)#sampler = torch.utils.data.RandomSampler(train_set, replacement=True), batch_size=batch_size)

# To obtain non-homogeneous batches: batches that are composed of one class.
batch_sample = hparams.batch_sample
if batch_sample == "sort" or batch_sample == "random_sort":
    batch_shuffle, targets_shuffle = sort_batches(train_loader, batch_size, device)
if batch_sample == "random_with_rpl" or batch_sample == "random_sort":
    data_list = list(train_loader)

###
# TRAINING
###

# To save the trajectory
weights_trajectory = []
loss_trajectory = []

net.train()
n_epoch = hparams.n_epoch

for epoch in range(n_epoch): # training loop
    train_correct = 0
    train_loss = 0
    print('Epoch {}'.format(epoch))
    
    # loop per epoch
    for i, (batch, targets) in enumerate(train_loader):
        if batch_sample == "random_with_rpl":
            j = np.random.randint(len(data_list))
            batch, targets = data_list[j]
            batch, targets = batch.to(device), targets.to(device)
        elif batch_sample == "sort":
            if i >= len(batch_shuffle):
                break
            batch, targets = batch_shuffle[i].to(device), targets_shuffle[i].to(device)
        elif batch_sample == "random_sort":
            j = np.random.randint(len(batch_shuffle))
            # print(j)
            batch, targets = batch_shuffle[j].to(device), targets_shuffle[j].to(device)
        else:
            batch, targets = batch.to(device), targets.to(device)
        
        if network_type == "CNN":
            batch_size = batch.size()[0]
            batch = batch.view((batch_size, 3, 32, 32))
        else:
            batch_size = batch.size()[0]
        output = net(batch)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #save the trajectory
        weights_i = [param.data.clone() for param in net.parameters()]
        weights_trajectory.append(weights_i)
        loss_trajectory.append(loss.item())

        pred = output.max(1, keepdim=True)[1]
        # print(pred)
        train_correct += pred.eq(targets.view_as(pred)).sum().item()
        train_loss += loss

        if i % 100 == 10: print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(
            train_loss / ((i+1) * batch_size_train), 100 * train_correct / ((i+1) * batch_size_train)))
        if alg == "GD" or alg == "NAG":print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(
            train_loss / ((i+1) * batch_size_train), 100 * train_correct / ((i+1) * batch_size_train)))

print('End of training.\n')
    
# === Test === ###
test_correct = 0
net.eval()

# loop, over whole test set
for i, (batch, targets) in enumerate(test_loader):
    batch = batch.to(device)
    if network_type == "CNN":
        batch_size = batch.size()[0]
        batch = batch.view((batch_size, 3, 32, 32))
    output = net(batch)
    targets = targets.to(device)
    pred = output.max(1, keepdim=True)[1]
    test_correct += pred.eq(targets.view_as(pred)).sum().item()
    
print('End of testing. Test accuracy {:.2f}%'.format(
    100 * test_correct / (len(test_loader) * batch_size_test)))

# Save the training loss
path_results = "results/"

# Save the training trajectory in a torch dictionary
dict_results = {
    "weights_trajectory" : weights_trajectory,
    "loss_trajectory" : loss_trajectory,
    "network_type"  : network_type,
    "batch_sample" : batch_sample,
    "n_epoch" : n_epoch,
    "batch_size" : batch_size,
    "momentum" : momentum,
    "lr" : lr,
    "test_accuracy" : 100 * test_correct / (len(test_loader) * 64),
}

save_name = path_results+network_type+'_n_epoch_'+str(n_epoch)+'_batch_'+batch_sample+'_alg_'+alg+'_'
torch.save(dict_results, save_name+'dict_results.pth')
print("Model save in the adress : "+save_name+'dict_results.pth')

plt.plot(loss_trajectory)
plt.xlabel("number of iterations")
plt.ylabel("Training Loss")
plt.savefig(save_name+"training_trajectory.png")