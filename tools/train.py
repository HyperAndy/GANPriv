import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from model.cgan import Generator, Discriminator
from config.config_cgan import Config
from utils import utils

# get configs
parser = Config().parser
configs = parser.parse_args()
print(configs)

# img_shape = (configs.channels, configs.img_size, configs.img_size)
shape = configs.size
cuda = True if torch.cuda.is_available() else False

# Initialize G and D models
generator = Generator(configs, shape)
discriminator = Discriminator(configs, shape)

# Loss function
adversarial_loss = nn.MSELoss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
data = utils.Data(os.path.join(configs.data_path, 'BC-TCGA-Normal.txt'),
                  os.path.join(configs.data_path, 'BC-TCGA-Tumor.txt'))
dataloader = DataLoader(data, batch_size=configs.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=configs.lr, betas=(configs.b1, configs.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=configs.lr, betas=(configs.b1, configs.b2))

FloatTensor = utils.floatTensor(cuda)
LongTensor = utils.longTensor(cuda)

# Training
for epoch in range(configs.n_epochs):
    for i, (x, labels) in enumerate(dataloader):  # imgs

        batch_size = x.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_x = Variable(x.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # Train Generator
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, configs.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, configs.n_classes, batch_size)))

        # Generate a batch of x
        gen_x = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_x, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Loss for real x
        validity_real = discriminator(real_x, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake x
        validity_fake = discriminator(gen_x.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if i % 10 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, configs.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
