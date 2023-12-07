# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, img_size, latent_size, n_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, img_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size * 8, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size * 4, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size * 2, img_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_size, n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, img_size, n_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
