import torch
from torch.utils.data import DataLoader, TensorDataset

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
torch.save({'data': data, 'labels': labels}, '/beegfs/jhermant/Momentum_Stochastic_GD/dataset/sphere/dataset_sphere.pth')