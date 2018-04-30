import argparse
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import itertools


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.img_size**2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, opt.latent_dim)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        latent = self.model(img_flat)
        return latent


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, opt.img_size**2),
            nn.Tanh()
        )

    def forward(self, noise):
        img_flat = self.model(noise)
        img = img_flat.view(img_flat.shape[0], opt.channels, opt.img_size, opt.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, latent):
        validity = self.model(latent)

        return validity


def sample_image(n_row, epoches_done, saved_img_dir="images"):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn((n_row ** 2, opt.latent_dim)).to(device)
    gen_imgs = decoder(z)
    save_image(gen_imgs, "{}/{}.png".format(saved_img_dir, epoches_done), 
               nrow=n_row, normalize=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, 
                        help='number of epochs of training (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='size of the batches (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='adam: learning rate (default: 0.001)')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: coefficient in computing running averages of gradient (default: 0.9)')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: coefficient in computing running averages of gradient\'s square (default: 0.999)')
    parser.add_argument('--latent_dim', type=int, default=100, 
                        help='dimensionality of latent space (default: 100)')
    parser.add_argument('--n_classes', type=int, default=10, 
                        help='number of classes for dataset (default: 10)')
    parser.add_argument('--img_size', type=int, default=32, 
                        help='size of each image dimension (default: 32)')
    parser.add_argument('--channels', type=int, default=1, 
                        help='number of image channels (default: 1)')
    parser.add_argument('--saved_img_dir', type=str, default='images', 
                        help='dir of saved images (default: images)')
    parser.add_argument('--saved_models', type=str, default='save_models',
                        help='dir of saved images (default: save_models)')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation (default: 8)')
    parser.add_argument('--disable_gpu', action='store_true', 
                        help='Flag whether to disable GPU')
    parser.add_argument('--disable_tensorboard', action='store_true', 
                        help='Flag whether to disable TensorBoard')
    parser.add_argument('--multi_gpu', action='store_true', 
                        help='Flag whether to use multiple GPUs.')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.saved_img_dir, exist_ok=True)
    os.makedirs(opt.saved_models, exist_ok=True)
    if not opt.disable_tensorboard:
        sys.path.append("..")
        from logger import Logger  #
        logger = Logger('logs')

    # Initialize generator and discriminator
    device = torch.device("cuda" if not opt.disable_gpu and 
                          torch.cuda.is_available() else "cpu")
    encoder       = Encoder().to(device)
    decoder       = Decoder().to(device)
    discriminator = Discriminator().to(device)
    if opt.multi_gpu and torch.cuda.device_count() > 1:
        encoder         = nn.DataParallel(encoder)
        decoder         = nn.DataParallel(decoder)
        discriminator   = nn.DataParallel(discriminator)
        opt.batch_size *= torch.cuda.device_count()
    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss functions
    adversarial_loss = nn.BCELoss().to(device)
    pixelwise_loss = nn.L1Loss().to(device)

    # Configure data loader
    os.makedirs('../../data/mnist', exist_ok=True)
    transform = transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])
    trainset = datasets.MNIST('../../data/mnist', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), 
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    sample_image(n_row=10, epoches_done=0, 
                 saved_img_dir=opt.saved_img_dir)
    for epoch in range(opt.n_epochs):

        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.full((batch_size, 1), 1.0).to(device)
            fake = torch.full((batch_size, 1), 0.0).to(device)

            # Configure input
            real_imgs = imgs.float().to(device)
            labels = labels.long().to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(encoded_imgs)
            g_loss =    0.001 * adversarial_loss(validity, valid) + \
                    0.999 * pixelwise_loss(decoded_imgs, real_imgs)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Sample noise as discriminator ground truth
            z = torch.randn((batch_size, opt.latent_dim)).to(device)
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()
            print ("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]"\
                   .format(epoch, opt.n_epochs - 1,
                           i, len(dataloader) - 1,
                           d_loss.item(), 
                           g_loss.item()))
        if not opt.disable_tensorboard:  # record training curves
            logger.scalar_summary("D loss: ", d_loss.item(), epoch)  #
            logger.scalar_summary("G loss: ", g_loss.item(), epoch)  #

        torch.save((encoder.state_dict(), decoder.state_dict(), discriminator.state_dict()), 
                    "{}/aae.th".format(opt.saved_models))
        sample_image(n_row=10, epoches_done=epoch, 
                     saved_img_dir=opt.saved_img_dir)
