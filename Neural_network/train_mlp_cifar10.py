import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define network structure 
net = nn.Sequential(nn.Linear(3 * 32 * 32, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), 
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 10)).to(device)
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
gradients_trajectory = []

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
        batch = batch.to(device)
        output = net(batch)
        targets = targets.to(device)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    output = net(batch)
    targets = targets.to(device)
    pred = output.max(1, keepdim=True)[1]
    test_correct += pred.eq(targets.view_as(pred)).sum().item()
    
print('End of testing. Test accuracy {:.2f}%'.format(
    100 * test_correct / (len(test_loader) * 64)))