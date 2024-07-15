import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t
import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm import tqdm
<<<<<<< HEAD
from models_architecture import create_mlp, create_cnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

network_type = "CNN"
# define network structure 

if network_type == "mlp":# MLP architecture
    net = create_mlp().to(device)

if network_type == "CNN":# Light CNN architecture
    net = create_cnn().to(device)

=======

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define network structure 
net = nn.Sequential(nn.Linear(3 * 32 * 32, 512), nn.ReLU(), nn.Linear(512, 128),  nn.ReLU(), nn.Linear(128, 10)).to(device)
>>>>>>> cbf3c5f946dc768792bef1c53a9407e149b194dd
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

# load data
to_tensor =  t.ToTensor()
normalize = t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
flatten =  t.Lambda(lambda x:x.view(-1))

transform_list = t.Compose([to_tensor, normalize, flatten])
train_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=True, transform=transform_list, download=True)
test_set = torchvision.datasets.CIFAR10(root='/beegfs/mrenaud/Momentum_Stochastic_GD/dataset', train=False, transform=transform_list, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)

# To save the trajectory
weights_trajectory = []
loss_trajectory = []

# === Train === ###
net.train()

<<<<<<< HEAD
n_epoch = 3
=======
n_epoch = 1
>>>>>>> cbf3c5f946dc768792bef1c53a9407e149b194dd

# train loop
for epoch in range(n_epoch):
    train_correct = 0
    train_loss = 0
    print('Epoch {}'.format(epoch))
    
    # loop per epoch 
    for i, (batch, targets) in enumerate(train_loader):
        batch = batch.to(device)
<<<<<<< HEAD
        if network_type == "CNN":
            batch_size = batch.size()[0]
            batch = batch.view((batch_size, 3, 32, 32))
=======
>>>>>>> cbf3c5f946dc768792bef1c53a9407e149b194dd
        output = net(batch)
        targets = targets.to(device)
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
<<<<<<< HEAD
    if network_type == "CNN":
        batch_size = batch.size()[0]
        batch = batch.view((batch_size, 3, 32, 32))
=======
>>>>>>> cbf3c5f946dc768792bef1c53a9407e149b194dd
    output = net(batch)
    targets = targets.to(device)
    pred = output.max(1, keepdim=True)[1]
    test_correct += pred.eq(targets.view_as(pred)).sum().item()
    
print('End of testing. Test accuracy {:.2f}%'.format(
    100 * test_correct / (len(test_loader) * 64)))
<<<<<<< HEAD

#Save the training loss
path_result = "results/"
=======
<<<<<<< HEAD
>>>>>>> cbf3c5f946dc768792bef1c53a9407e149b194dd

plt.plot(loss_trajectory)
plt.xlabel("number of iterations")
plt.ylabel("Loss")
<<<<<<< HEAD
plt.savefig(path_result+"Loss trajectory.png")

dict_results = {
    "weights_trajectory" : weights_trajectory,
    "loss_trajectory" : loss_trajectory
}

torch.save(dict_results, path_result+'dict_results.pth')
=======
plt.savefig("Loss trajectory.png")

print("Number of iterations = {}".format(len(weights_trajectory)))

post_process_loader = torch.utils.data.DataLoader(train_set, batch_size=128)

for k in range(len(weights_trajectory)):
    print("Iteration {}".format(10*k))
    x = weights_trajectory[10*k]
    for j, param in enumerate(net.parameters()):
        param.data = x[j]
    gradient_list = []

    for i, (batch, targets) in enumerate(post_process_loader):
        batch = batch.to(device)
        output = net(batch)
        targets = targets.to(device)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        #save gardients
        gradient_i = np.array([])
        for param in net.parameters():
            gradient_i = np.concatenate((gradient_i,np.array(param.grad.detach().cpu().numpy().reshape(-1))))
        gradient_list.append(gradient_i)

    gradient_list = np.array(gradient_list)

    norm = 0
    sum_gradient = np.zeros(len(gradient_list[0]))
    for j in tqdm(range(len(gradient_list))):
        norm += np.sum(np.array(gradient_list[j])**2)
        sum_gradient += gradient_list[j]

    scalar_prod = (1/2) * (np.sum(np.array(sum_gradient)**2) - norm)
    racoga = scalar_prod/norm
    print("RACOGA = {}".format(racoga))
>>>>>>> cbf3c5f946dc768792bef1c53a9407e149b194dd
