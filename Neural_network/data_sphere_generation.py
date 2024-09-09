import torch
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
def points_sur_sphere_torch(d, n_points):
    """
    Generate points uniformly distributed on the sphere.
    
    :param d: Dimension of the sphere.
    :param n_points: Number of points to generate.
    :return: Tensor whose shape is (n_points, d) each line is a point on the sphere.
    """
    # Étape 1: Générer des points dans un espace de dimension d suivant une distribution normale
    points = torch.randn(n_points, d)
    
    # Étape 2: Normaliser les points pour qu'ils soient sur la sphère
    points = points / points.norm(dim=1, keepdim=True)
    
    return points


d = 32*32*3        # Dimension 
n_points = 60000  # Number of points to generate

data = points_sur_sphere_torch(d, n_points)
# Very simple label: one hemisphere is label "0", the other is label "1"
labels = torch.zeros(n_points,dtype = torch.long)
labels[torch.where((data[:,0]<0))]  =1

# Save dataset
dataset = TensorDataset(data, labels)
train_size = 50000
test_size =  10000
    # Split dataset between training set and test set
torch.manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
torch.manual_seed(42)
train_labels, test_labels = torch.utils.data.random_split(labels, [train_size, test_size])
torch.save({'data': train_dataset, 'labels': train_labels}, '/beegfs/jhermant/Momentum_Stochastic_GD/dataset/sphere/train_dataset_sphere.pth')
torch.save({'data': test_dataset, 'labels': test_labels}, '/beegfs/jhermant/Momentum_Stochastic_GD/dataset/sphere/test_dataset_sphere.pth')