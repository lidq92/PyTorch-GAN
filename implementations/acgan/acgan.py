import argparse
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(128 * ds_size ** 2, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(128 * ds_size ** 2, opt.n_classes),
                                        nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


def sample_image(n_row, epoches_done, saved_img_dir="images"):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn((n_row ** 2, opt.latent_dim)).to(device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = torch.stack([torch.arange(end=n_row) for _ in range(n_row)]).reshape((n_row ** 2)).long().to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs, "{}/{}.png".format(saved_img_dir, epoches_done), nrow=n_row, normalize=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate (default: 0.0002)')
    parser.add_argument('--b1', type=float, default=0.5,
                        help='adam: coefficient in computing running averages of gradient (default: 0.5)')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: coefficient in computing running averages of gradient\'s square (default: 0.999)')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space (default: 100)')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset (default: 10)')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension (default: 32)')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels (default: 1)')
    parser.add_argument('--saved_img_dir', type=str, default='images', help='dir of saved images (default: images)')
    parser.add_argument('--saved_models', type=str, default='save_models',
                        help='dir of saved images (default: save_models)')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation (default: 8)')
    parser.add_argument('--disable_gpu', action='store_true', help='Flag whether to disable GPU')
    parser.add_argument('--enable_tensorboard', action='store_false', help='Flag whether to enable TensorBoard')
    parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use multiple GPUs.')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.saved_img_dir, exist_ok=True)
    os.makedirs(opt.saved_models, exist_ok=True)
    if opt.enable_tensorboard:
        sys.path.append("..")
        from logger import Logger  #
        logger = Logger('logs')

    # Initialize generator and discriminator
    device = torch.device("cuda" if not opt.disable_gpu and torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    if opt.multi_gpu and torch.cuda.device_count() > 1:
        generator       = nn.DataParallel(generator)
        discriminator   = nn.DataParallel(discriminator)
        opt.batch_size *= torch.cuda.device_count()
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss functions
    adversarial_loss = nn.BCELoss().to(device)
    auxiliary_loss = nn.CrossEntropyLoss().to(device)

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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):

        sample_image(n_row=10, epoches_done=epoch, saved_img_dir=opt.saved_img_dir)

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
            # Sample noise and labels as generator input
            z = torch.randn((batch_size, opt.latent_dim)).to(device)
            gen_labels = torch.randint(0, opt.n_classes, (batch_size,)).long().to(device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) +
                            auxiliary_loss(pred_label, gen_labels))
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) +
                           auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) +
                           auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Calculate discriminator accuracy
            pred = torch.cat((real_aux.detach(), fake_aux.detach()))
            gt = torch.cat((labels, gen_labels))
            d_acc = torch.mean((torch.argmax(pred, dim=1) == gt).float().to("cpu"))
        if opt.enable_tensorboard:  # record training curves
            logger.scalar_summary("D loss: ", d_loss.item(), epoch)  #
            logger.scalar_summary("G loss: ", g_loss.item(), epoch)  #
            logger.scalar_summary("D acc: ", 100 * d_acc, epoch)  #
        else:
            print ("[Epoch {}/{}] [Batch {}/{}] [D loss: {}, acc: {}%%] [G loss: {}]".format(epoch, opt.n_epochs,
                                                                                             i, len(dataloader),
                                                                                             d_loss.item(), 100 * d_acc,
                                                                                             g_loss.item()))

        torch.save(generator.state_dict(), "{}/generator.th".format(opt.saved_models))
