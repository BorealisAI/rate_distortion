# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang and Yanshuai Cao

#!/usr/bin/env python3

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class sheldonMLPGenerator(nn.Module):

    def __init__(self,
                 hdim=1024,
                 img_dims=(1, 28, 28),
                 zdim=100,
                 spectral_norm=False,
                 flat_out=True):
        super(sheldonMLPGenerator, self).__init__()

        self.img_dims = img_dims
        out_len = int(np.prod(img_dims))
        self.de1 = nn.Linear(zdim, hdim)
        self.de2 = nn.Linear(hdim, hdim)
        self.de3 = nn.Linear(hdim, hdim)
        self.out = nn.Linear(hdim, out_len)

        self.nl = torch.tanh
        self.default_flat_out = flat_out

    def forward(self, z, flat=None):
        h = self.de1(z)
        h = self.nl(h)

        h = self.de2(h)
        h = self.nl(h)

        h = self.de3(h)
        h = self.nl(h)

        out = self.out(h)
        out = torch.sigmoid(out)

        if ((flat is None and not self.default_flat_out) or
            (flat is not None and not flat)):
            out = out.reshape(z.size(0), *self.img_dims)

        return out

    def decode(self, z):
        return self.forward(z, flat=True), self.x_logvar


class sheldonMLPGenerator3(nn.Module):

    def __init__(self,
                 hdim=512,
                 img_dims=(1, 28, 28),
                 zdim=100,
                 spectral_norm=False,
                 flat_out=True):
        super(sheldonMLPGenerator3, self).__init__()

        self.img_dims = img_dims
        out_len = int(np.prod(img_dims))
        self.de1 = nn.Linear(zdim, hdim)
        self.de2 = nn.Linear(hdim, hdim)
        self.out = nn.Linear(hdim, out_len)

        self.nl = torch.tanh
        self.default_flat_out = flat_out

    def forward(self, z, flat=None):
        h = self.de1(z)
        h = self.nl(h)

        h = self.de2(h)
        h = self.nl(h)

        out = self.out(h)
        out = torch.sigmoid(out)

        if ((flat is None and not self.default_flat_out) or
            (flat is not None and not flat)):
            out = out.reshape(z.size(0), *self.img_dims)

        return out

    def decode(self, z):
        return self.forward(z, flat=True), self.x_logvar


class sheldonMLPGenerator2(nn.Module):

    def __init__(self,
                 hdim=512,
                 img_dims=(1, 28, 28),
                 zdim=100,
                 spectral_norm=False,
                 flat_out=True):
        super(sheldonMLPGenerator2, self).__init__()

        self.img_dims = img_dims
        out_len = int(np.prod(img_dims))
        self.de1 = nn.Linear(zdim, hdim)
        self.out = nn.Linear(hdim, out_len)

        self.nl = torch.tanh
        self.default_flat_out = flat_out

    def forward(self, z, flat=None):
        h = self.de1(z)
        h = self.nl(h)
        out = self.out(h)
        out = torch.sigmoid(out)

        if ((flat is None and not self.default_flat_out) or
            (flat is not None and not flat)):
            out = out.reshape(z.size(0), *self.img_dims)

        return out

    def decode(self, z):
        return self.forward(z, flat=True), self.x_logvar


class DCGANGenerator(nn.Module):

    def __init__(self,
                 ngf=64,
                 size=32,
                 zdim=100,
                 nc=3,
                 ngpu=1,
                 nm='bn',
                 spectral_norm=False,
                 n_convs=4):

        super(DCGANGenerator, self).__init__()
        self.ngpu = ngpu
        self.size = size
        self.ngf = ngf
        self.spectral_norm = spectral_norm
        self.n_convs = n_convs
        self.nm = nm

        proj_size = size
        self.proj_size = int(proj_size)
        self.zdim = zdim

        layers = []
        layers.append(nn.Tanh())

        conv_bias = False
        if self.nm == 'bn':
            nm = nn.BatchNorm2d
        elif self.nm == 'in':
            nm = nn.InstanceNorm2d
        elif self.nm.lower() in ('none',):
            nm = lambda x: None
            conv_bias = True
        elif self.nm.lower() in ('ln',):
            nm = nn.LayerNorm
            conv_bias = True
        else:
            raise ValueError()

        nchannels = ngf
        print("Last layer, in channels: {}, out channels: {}".format(
            nchannels, nc))
        layers.append(
            nn.ConvTranspose2d(nchannels, nc, 4, 2, 1, bias=conv_bias))
        layers.append(nn.ReLU(True))
        if nm == nn.LayerNorm:
            layers.append(nm(nchannels, proj_size, proj_size))
        else:
            layers.append(nm(nchannels))

        proj_size = int(np.ceil(proj_size / 2.))

        for idx in range(self.n_convs - 1):

            nchannels *= 2
            print("{} th layer, in channels: {}, out channels: {}".format(
                self.n_convs - 1 - idx, nchannels, nchannels // 2))
            layers.append(
                nn.ConvTranspose2d(
                    nchannels, nchannels // 2, 4, 2, 1, bias=conv_bias))
            layers.append(nn.ReLU(True))
            if nm == nn.LayerNorm:
                layers.append(nm(nchannels, proj_size, proj_size))
            else:
                layers.append(nm(nchannels))

            proj_size = int(np.ceil(proj_size / 2.))

        layers = [l for l in layers if l is not None]

        self.ini_channels = nchannels
        self.proj_size = proj_size
        self.proj = nn.Linear(zdim, nchannels * self.proj_size**2, bias=False)

        layers = layers[::-1]

        if self.spectral_norm:
            for idx in range(len(layers)):
                if isinstance(layers[idx], nn.ConvTranspose2d):
                    layers[idx] = nn.utils.spectral_norm(layers[idx])

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

        # DCGANGenerator(
        # (proj): Linear(in_features=100, out_features=4096, bias=False)
        # (main): Sequential(
        #     (0): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (1): ReLU(inplace)
        #     (2): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (4): ReLU(inplace)
        #     (5): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (7): ReLU(inplace)
        #     (8): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (10): ReLU(inplace)
        #     (11): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (12): Tanh()
        # )
        # )

    def forward(self, input):

        h = self.proj(input)
        h = h.view(h.size(0), self.ini_channels, self.proj_size, self.proj_size)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, h, range(self.ngpu))
        else:
            output = self.main(h)

        if output.size(2) > self.size:
            output = output[:, :, :self.size, :self.size]
        return output

    def decode(self, z):
        return self.forward(z).view(z.size(0), -1), self.x_logvar
