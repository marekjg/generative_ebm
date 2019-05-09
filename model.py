import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample=False):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, mid_channels, 3,
                                             stride=2 if downsample else 1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(mid_channels, mid_channels, 3, padding=1))
        if downsample:
            self.avg_pool = nn.AvgPool2d(2, ceil_mode=True)
        if in_channels != mid_channels:
            self.conv1x1 = spectral_norm(nn.Conv2d(in_channels, mid_channels, 1))

        self.in_out_match = (in_channels == mid_channels)
        self.downsample = downsample

    def forward(self, x):
        h = self.conv1(x).relu()
        h = self.conv2(h).relu()
        if self.downsample:
            x = self.avg_pool(x)
        if not self.in_out_match:
            x = self.conv1x1(x)

        return (h + x).relu()


class MnistEnergyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 128, 3, padding=1)),
            ResBlock(128, 128, True),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256, True),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256, True),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )

    def forward(self, x):
        h = self.resnet(x).view(-1, 256 * 4 * 4)
        energy = h.sum(dim=1)
        return energy


class MnistCondEnergyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 28*28)
        self.resnet = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 128, 3, padding=1)),
            ResBlock(128, 128, True),
            ResBlock(128, 128),
            ResBlock(128, 256, True),
            ResBlock(256, 256),
            ResBlock(256, 256, True),
            ResBlock(256, 256)
        )
        self.fc = spectral_norm(nn.Linear(256 * 4 * 4, 1))

    def forward(self, x, y):
        y_embed = self.embed(y).view(-1, 1, 28, 28)
        h = torch.cat((x, y_embed), dim=1)
        h = self.resnet(h).view(-1, 256 * 4 * 4)
        energy = self.fc(h)
        return energy
