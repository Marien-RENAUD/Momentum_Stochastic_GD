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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument('--network_type', type=str, default = "CNN")
hparams = parser.parse_args()


# define network structure 
network_type = hparams.network_type

if network_type == "mlp":# MLP architecture
    net = create_mlp().to(device)

if network_type == "CNN":# Light CNN architecture
    net = create_cnn().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

# load data
to_tensor =  t.ToTensor()
normalize = t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
flatten =  t.Lambda(lambda x:x.view(-1))

transform_list = t.Compose([to_tensor, normalize, flatten])
train_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=True, transform=transform_list, download=True)
test_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=False, transform=transform_list, download=True)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# To obtain non-homogeneous batches: batches that are composed of one class.
non_homogeneous = True
if non_homogeneous:
    batch_sort = [[] for i in range(10)]
    targets_sort = [[] for i in range(10)]
    num_im = 0
    for i, (batch, targets) in tqdm(enumerate(train_loader)):
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
    batch_shuffle = torch.tensor(batch_shuffle)
    targets_shuffle = torch.tensor(targets_shuffle)

# To save the trajectory
weights_trajectory = []
loss_trajectory = []

# === Train === ###
net.train()

n_epoch = 3

# train loop
for epoch in range(n_epoch):
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
path_result = "results/"

plt.plot(loss_trajectory)
plt.xlabel("number of iterations")
plt.ylabel("Training Loss")
plt.savefig(path_result+"Training_trajectory.png")

# Save the training trajectory in a torch dictionary
dict_results = {
    "weights_trajectory" : weights_trajectory,
    "loss_trajectory" : loss_trajectory
}

torch.save(dict_results, path_result+'dict_results.pth')
