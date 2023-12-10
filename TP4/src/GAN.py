import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, img_size, latent_size, n_channels):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.n_channels = n_channels

        self.model = nn.Sequential(
            nn.Linear(latent_size, img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size * 2, img_size * 4),
            nn.BatchNorm1d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size * 4, img_size * 8),
            nn.BatchNorm1d(img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size * 8, img_size * 16),
            nn.BatchNorm1d(img_size * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size * 16, int(np.prod((n_channels, img_size, img_size)))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), self.n_channels, self.img_size, self.img_size)


class Discriminator(nn.Module):
    def __init__(self, img_size, n_channels):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.n_channels = n_channels

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((n_channels, img_size, img_size))), img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(img_size * 4, img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(img_size * 2, img_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(img_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)
