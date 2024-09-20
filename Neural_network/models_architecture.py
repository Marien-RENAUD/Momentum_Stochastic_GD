import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t

def create_mlp():
    """
    MLP net
    """
    return nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.ReLU(), nn.Linear(128, 64),  nn.ReLU(), nn.Linear(64, 2))

def create_cnn():
    """
    CNN net without batch normalization
    """
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

#-- Test batch normalization --#

def create_cnn_bn():
    """
    CNN net with batch normalization
    """
    net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),

    nn.Flatten(),
    nn.Linear(128, 10)
    )
    return net
