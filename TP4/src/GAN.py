import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, normalization_choice):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalization_choice > 1 and normalize:
                if normalization_choice % 2 == 0:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                else:  # normalization_choice =1 or 3
                    layers.append(
                        nn.InstanceNorm1d(out_feat, 0.8)
                    )  # out_feat size -> 64
                # layers.append(spectral_norm(nn.BatchNorm1d(out_feat, 0.8))) useless?
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, normalization_choice):
        super(Discriminator, self).__init__()
        if normalization_choice == 2 or normalization_choice > 3:
            self.model = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(int(np.prod(img_shape)), 512)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(512, 256)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(256, 1)),
                nn.Sigmoid(),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
