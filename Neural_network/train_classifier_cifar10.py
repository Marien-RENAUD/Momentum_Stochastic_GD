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
import itertools

# Define Parser arguments
parser = ArgumentParser()
parser.add_argument('--network_type', type=str, default = "CNN")
parser.add_argument('--non_homogeneous', type=bool, default = False)
parser.add_argument('--device', type=int, default = 0)
parser.add_argument('--n_epoch', type=int, default = 5)
hparams = parser.parse_args()

device = torch.device('cuda:'+str(hparams.device) if torch.cuda.is_available() else 'cpu')

# define network structure 
network_type = hparams.network_type

if network_type == "mlp":# MLP architecture
    net = create_mlp().to(device)

if network_type == "CNN":# Light CNN architecture
    net = create_cnn().to(device)

criterion = nn.CrossEntropyLoss()
momentum = 0.9
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum)

###
# DATA PRE-PROCESSING
###

# load data
to_tensor =  t.ToTensor()
normalize = t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
flatten =  t.Lambda(lambda x:x.view(-1))

transform_list = t.Compose([to_tensor, normalize, flatten])
train_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=True, transform=transform_list, download=True)
test_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=False, transform=transform_list, download=True)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)#, sampler = torch.utils.data.RandomSampler(train_set, replacement=True), batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

def get_batch_direct(dataloader, i):
    """
    Get the batch of index i of the dataloader
    """
    return next(itertools.islice(dataloader, i, None))

# To obtain non-homogeneous batches: batches that are composed of one class.
non_homogeneous = hparams.non_homogeneous
if non_homogeneous:
    batch_sort = [[] for i in range(10)]
    targets_sort = [[] for i in range(10)]
    num_im = 0
    for _ in tqdm(range(len(train_loader))):
        i = np.random.randint(len(train_loader))
        batch, targets = get_batch_direct(train_loader, i)
        index_sort = np.argsort(targets)
        batch_sort_i = batch[index_sort]
        targets_sort_i = targets[index_sort]
        for j in range(len(targets_sort_i)):
            num_im += 1
            batch_sort[targets_sort_i[j]].append(batch_sort_i[j].numpy())
            targets_sort[targets_sort_i[j]].append(int(targets_sort_i[j]))

    batch_sort_2 = []
    targets_sort_2 = []
    for k in range(10):
        batch_sort_2 = batch_sort_2 + batch_sort[k]
        targets_sort_2 = targets_sort_2 + targets_sort[k]
    batch_shuffle = []
    targets_shuffle = []
    for i in range(len(targets_sort_2)//batch_size):
        batch_shuffle.append(batch_sort_2[i*batch_size:(i+1)*batch_size])
        targets_shuffle.append(targets_sort_2[i*batch_size:(i+1)*batch_size])
    batch_shuffle = np.array(batch_shuffle)
    batch_shuffle = torch.tensor(batch_shuffle).to(device)
    targets_shuffle = torch.tensor(targets_shuffle).to(device)

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
        if non_homogeneous:
            if i >= len(batch_shuffle):
                break
            batch = batch_shuffle[i].to(device)
        else:
            batch = batch.to(device)
        if network_type == "CNN":
            batch_size = batch.size()[0]
            batch = batch.view((batch_size, 3, 32, 32))
        output = net(batch)
        if non_homogeneous:
            targets = targets_shuffle[i].to(device)
        else:
            targets =targets.to(device)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #save the trajectory
        weights_i =[param.data.clone() for param in net.parameters()]
        weights_trajectory.append(weights_i)
        loss_trajectory.append(loss.item())

        pred = output.max(1, keepdim=True)[1]
        train_correct += pred.eq(targets.view_as(pred)).sum().item()
        train_loss += loss

        if i % 100 == 10: print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(
            train_loss / ((i+1) * 64), 100 * train_correct / ((i+1) * 64)))
        
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
    100 * test_correct / (len(test_loader) * 64)))


# Save the training loss
path_results = "results/"

plt.plot(loss_trajectory)
plt.xlabel("number of iterations")
plt.ylabel("Training Loss")
plt.savefig(path_results+"Training_trajectory.png")

# Save the training trajectory in a torch dictionary
dict_results = {
    "weights_trajectory" : weights_trajectory,
    "loss_trajectory" : loss_trajectory,
    "network_type"  : network_type,
    "non_homogeneous" : non_homogeneous,
    "n_epoch" : n_epoch,
    "batch_size" : batch_size,
    "momentum" : momentum,
    "lr" : lr,
    "test_accuracy" : 100 * test_correct / (len(test_loader) * 64),
}

save_name = path_results+network_type+'_n_epoch_'+str(n_epoch)+'_dict_results.pth'
if non_homogeneous:
    save_name = path_results+network_type+'_n_epoch_'+str(n_epoch)+'_nonhomogeneous_dict_results.pth'
torch.save(dict_results, save_name)