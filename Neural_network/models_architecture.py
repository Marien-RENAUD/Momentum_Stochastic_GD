import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t

def create_mlp():
    return nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.ReLU(), nn.Linear(128, 64),  nn.ReLU(), nn.Linear(64, 2))

def create_cnn():
    net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),

    nn.Flatten(),
    nn.Linear(128, 10)
    )
    return net

