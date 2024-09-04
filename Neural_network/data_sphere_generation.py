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

# Exemple d'utilisation
d = 32*32*3        # Dimension de la sphère
n_points = 60000  # Nombre de points à générer

data = points_sur_sphere_torch(d, n_points)
labels = torch.zeros(n_points,dtype = torch.long)
# labels[torch.where(data[:,:16*32*3].sum(axis=1) < 0)] = 1
labels[torch.where((data[:,0]<0))]  =1
# labels[torch.where((data[:,0]>0)*(data[:,100]<0))]  =2
# labels[torch.where((data[:,0]>0)*(data[:,100]>0))]  =3
# labels[:16*32*3]=1
dataset = TensorDataset(data, labels)
torch.save({'data': data, 'labels': labels}, '/beegfs/jhermant/Momentum_Stochastic_GD/dataset/sphere/dataset_sphere.pth')