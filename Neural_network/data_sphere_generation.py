import torch
from torch.utils.data import DataLoader, TensorDataset
import os

torch.manual_seed(42)

def sphere_uniform(d, n_points):
    points = torch.randn(n_points, d)
    points = points / points.norm(dim=1, keepdim=True)
    
    return points

d = 32*32*3        # Dimension 
n_points = 60000  # Number of points to generate

data = sphere_uniform(d, n_points)

# Very simple label: one hemisphere is label "0", the other is label "1"
labels = torch.zeros(n_points,dtype = torch.long)
labels[torch.where((data[:,0]<0))]  =1

# Save dataset
dataset = TensorDataset(data, labels)
train_size = 50000
test_size =  10000
# Split dataset between training set and test set
torch.manual_seed(42)
train_data = data[:50000]
train_labels = labels[:50000] 
test_data = data[50000:]
test_labels = labels[50000:]
path = "../dataset"
path = os.path.join(path, "sphere")
if not os.path.exists(path):
    os.mkdir(path)
path = "../dataset/sphere/"
torch.save({'data': train_data, 'labels': train_labels}, path + 'train_dataset_sphere.pth')
torch.save({'data': test_data, 'labels': test_labels}, path + 'test_dataset_sphere.pth')
