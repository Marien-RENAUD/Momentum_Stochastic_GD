import itertools
import numpy as np
import torch

def get_batch_direct(dataloader, i):
    """
    Get the batch of index i of the dataloader
    """
    return next(itertools.islice(dataloader, i, None))

def sort_batches(train_loader, batch_size, device):
    batch_sort = [[] for i in range(10)]
    targets_sort = [[] for i in range(10)]
    num_im = 0
    for i, (batch, targets) in enumerate(train_loader):
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