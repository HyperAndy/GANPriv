import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, configs, shape):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(configs.n_classes, configs.n_classes)
        self.shape = shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(configs.latent_dim + configs.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and data to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        input = self.model(gen_input)
        input = input.view(input.size(0), -1) # resize
        return input


class Discriminator(nn.Module):
    def __init__(self, configs, shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(configs.n_classes, configs.n_classes)

        self.model = nn.Sequential(
            nn.Linear(configs.n_classes + int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, input, labels):
        # Concatenate label embedding and data to produce input
        d_in = torch.cat((input.view(input.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


