import logging
from collections import deque

import visdom

import numpy as np
import torch
from torch import optim
import torchvision

from model import MnistEnergyNN, MnistCondEnergyNN, SimpleEnergyNN, SimpleCondEnergyNN


def initialize_replay_buffer(replay_buffer, n=32):
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n, 1))
    for i in range(n):
        replay_buffer.append((x[i], y[i]))


def sample_from_replay_buffer(replay_buffer, n=32):
    x_negative = torch.rand(n, 1, 28, 28)
    y_negative = torch.randint(0, 10, (n, 1))
    idx_samples = []
    n_samples = np.random.binomial(n, 0.95)
    idx_samples = np.random.choice(range(len(replay_buffer)),
                                   size=n_samples, replace=False)
    for i, idx in enumerate(idx_samples):
        x_negative[i] = replay_buffer[idx][0]
        y_negative[i] = replay_buffer[idx][1]
    return x_negative, y_negative


def update_replay_buffer(replay_buffer, batch):
    x, y = batch
    for i in range(x.shape[0]):
        replay_buffer.append((x[i], y[i]))


def langevin_rollout(x_negative, y_negative, energy_nn, step_size, lambda_var, K):
    x_negative = x_negative.requires_grad_(True)
    for k in range(K):
        energy_nn(x_negative, y_negative).sum().backward()
        with torch.no_grad():
            x_negative.grad.clamp_(-0.01, 0.01)
            x_negative -= step_size * x_negative.grad / 2
            x_negative += lambda_var * torch.randn(*x_negative.shape).to(device)
            x_negative.clamp_(0, 1)
        x_negative.grad.zero_()
    return x_negative.requires_grad_(False)

def loss_fn(x_positive, y_positive, x_negative, y_negative, energy_nn, alpha):
    x_positive_energy = energy_nn(x_positive, y_positive)
    x_negative_energy = energy_nn(x_negative, y_negative)
    loss_l2 = alpha * (x_positive_energy.pow(2) + x_negative_energy.pow(2)).mean()
    loss_ml = (x_positive_energy - x_negative_energy).mean()
    return loss_l2 + loss_ml, loss_l2.item(), loss_ml.item()

device = 'cuda:0'

K = 60
step_size = 20
lambda_var = 0.01
alpha = 1

lr = 0.001
batch_size = 128

energy_nn = MnistCondEnergyNN().to(devie)
optimizer = optim.Adam(energy_nn.parameters())
replay_buffer = deque(maxlen=1000)
initialize_replay_buffer(replay_buffer, n=batch_size)
data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', transform=torchvision.transforms.ToTensor(), download=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2)

viz = visdom.Visdom()
logging.basicConfig(filename='mnist.log', level=logging.DEBUG)

it = 0
while True:
    x_positive, y_positive = next(iter(data_loader))
    x_negative, y_negative = sample_from_replay_buffer(replay_buffer, n=batch_size)
    x_positive = x_positive.to(device)
    y_positive = y_positive.to(device)
    x_negative = x_negative.to(device)
    y_negative = y_negative.to(device)

    langevin_rollout(x_negative, y_negative, energy_nn, step_size, lambda_var, K)
    update_replay_buffer(replay_buffer, (x_negative.to('cpu'), y_negative.to('cpu')))
    if it % 1000 == 0:
        viz.images(x_negative)

    optimizer.zero_grad()
    loss, loss_l2, loss_ml = loss_fn(x_positive, y_positive, x_negative, y_negative, energy_nn, alpha)
    logging.info('%f,%f' % (loss_l2, loss_ml))
    loss.backward()
    torch.nn.utils.clip_grad_value_(energy_nn.parameters(), 0.01)
    optimizer.step()

    it += 1
