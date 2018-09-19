from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from mnist_VAE import mnist_VAE
from mnist_VAE_conv import mnist_VAE_conv
from YCB_VAE import YCB_VAE
import ycb_loader
import util
import visualize as vis
import numpy as np


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--dataset-folder', type=str, required=True, default='', metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


img_res = (64, 64)

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    print("Using CUDA")
else:
    print("Not using CUDA")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = ycb_loader.DataLoader(args.dataset_folder + 'train_small/', batch_size=args.batch_size, img_res=img_res)
test_loader = ycb_loader.DataLoader(args.dataset_folder + 'test_small/',  batch_size=args.batch_size, img_res=img_res)

print('Lenght of training dataset: {}'.format(len(train_loader.dataset)))
print('Lenght of test dataset: {}'.format(len(test_loader.dataset)))


model = YCB_VAE(batch_size=args.batch_size, img_res=img_res).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    x_length = x.shape[1] * x.shape[2] * x.shape[3]
    BCE = F.binary_cross_entropy(recon_x.view(-1, x_length), x.view(-1, x_length), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def loss_function1(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    #print('BCE: {}'.format(BCE))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print('LogVar: {}'.format(logvar))
    #print('KLD: {}'.format(KLD))
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        #data = util.add_noise_torch(data, blackout_prob=0.25)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function1(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            #data = util.add_noise_torch(data, blackout_prob=0.25)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function1(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 4, 64, 64)[:n]])
                #save_image(comparison.cpu(),
                #         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            if i % args.log_interval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(test_loader.dataset),
                    100. * i / len(test_loader),
                    test_loss / len(data)))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(args.batch_size, model.latent_space_size).to(device)
        sample = model.decode(sample).cpu()
        sample = sample[:, 0:3, :, :]
        sample *= 255
        sample = sample.int()
        save_image(sample.view(args.batch_size, 3, img_res[0], img_res[1]).int(),
                   'results/sample_' + str(epoch) + '.png')