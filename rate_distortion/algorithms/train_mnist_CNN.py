# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import torch
import numpy as np
from ..utils.experiment_utils import note_taking, save_checkpoint, sample_images, get_chechpoint_path, load_checkpoint
from torchvision.utils import save_image
from ..data.load_data import load_training_data
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from ..models.registry import get_cnn_model


def train(args, model, device, train_loader, optimizer, epoch, hparams):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(
            device=hparams.device, dtype=hparams.tensor_type), target.to(
                device=hparams.device, dtype=hparams.tensor_type)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(args, model, device, test_loader, hparams):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(
                device=hparams.device, dtype=hparams.tensor_type), target.to(
                    device=hparams.device, dtype=hparams.tensor_type)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= 10000

    note_taking(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, 10000, 100. * correct / 10000))


def get_mnist_representation(args, hparams):
    device = hparams.device
    model = get_cnn_model(hparams).to(device)
    checkpoint_path = os.path.join(hparams.messenger.checkpoint_dir_rep,
                                   "mnist_cnn_checkpoint.pth")
    if hparams.rep_train_first:
        train_loader, test_loader = load_training_data(hparams)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, hparams)
            test(args, model, device, test_loader, hparams)

        torch.save({
            "state_dict": model.state_dict(),
        }, checkpoint_path)

    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint = torch.load(checkpoint_path)
        note_taking(
            "loaded representation from checkpoint: {}".format(checkpoint_path))

    return model
