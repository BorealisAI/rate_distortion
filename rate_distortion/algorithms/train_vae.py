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
from tqdm import tqdm
import os


def train_epoch(epoch, model, train_loader, hparams, optimizer, task_params):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader, 1):
        data = data.to(device=hparams.device, dtype=hparams.tensor_type)
        optimizer.zero_grad()
        _, elbo, _, _, _, _, _ = model(data)
        loss = -elbo
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    note_taking(
        'Training epoch: {} Average negative elbo on train set: {}'.format(
            epoch, train_loss / batch_idx))

    try:
        num_samples = len(train_loader.dataset)
    except:
        num_samples = len(train_loader) * task_params.batch_size
    return train_loss, num_samples


def test_epoch(epoch,
               model,
               test_loader,
               hparams,
               task_params,
               do_logging,
               best=False):
    model.eval()
    elbos = list()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader, 1):
            data = data.to(device=hparams.device, dtype=hparams.tensor_type)
            recon_batch, elbo, _, _, _, _, _ = model(data)
            elbos.append(elbo.cpu().numpy().item())
            if do_logging:
                if epoch is not None:
                    if i == 1:
                        if hparams.dataset.data_name == "cifar10":
                            recon_batch = recon_batch / 2 + 0.5
                        n = min(data.size(0), 8)
                        comparison = torch.cat([
                            data[:n],
                            recon_batch.view(task_params.batch_size,
                                             hparams.dataset.input_dims[0],
                                             hparams.dataset.input_dims[1],
                                             hparams.dataset.input_dims[2])[:n]
                        ])
                        save_image(
                            comparison.cpu(),
                            hparams.messenger.image_dir +
                            ('best_' if best else '') + 'reconstruction_' +
                            str(epoch) + '.png',
                            nrow=n)
            if i == hparams.n_test_batch:
                break

    test_loss = -np.mean(elbos)
    note_taking('Test loss: {} over {} batches'.format(test_loss, float(i)))
    return test_loss


def train_and_test(test_loss_list, model, optimizer, writer, hparams):
    train_loader, test_loader = load_training_data(hparams)

    for epoch in tqdm(range(hparams.epoch, hparams.model_train.epochs + 1)):
        train_loss, steps = train_epoch(epoch, model, train_loader, hparams,
                                        optimizer, hparams.model_train)
        hparams.step += steps
        hparams.epoch += 1
        writer.add_scalar('train_time/train_loss', train_loss, epoch)
        do_save_ckpt = epoch % hparams.checkpointing_freq == 0
        do_logging = epoch % hparams.train_print_freq == 0
        with torch.no_grad():
            test_loss = test_epoch(epoch, model, test_loader, hparams,
                                   hparams.model_train, do_logging)
            writer.add_scalar('train_time/test_loss', test_loss, epoch)
            test_loss_list.append(test_loss)
            if epoch > hparams.start_checkpointing:
                if len(test_loss_list) != 0:
                    if test_loss <= min(test_loss_list[:-1]):
                        note_taking(
                            'new min test loss is {} at epoch {}'.format(
                                test_loss, epoch))

                        checkpoint_path = os.path.join(
                            hparams.messenger.checkpoint_dir, "best.pth")

                        save_checkpoint(
                            checkpoint_path=checkpoint_path,
                            optimizer=optimizer,
                            save_optimizer_state=True,
                            model=model,
                            hparams=hparams,
                            test_loss_list=test_loss_list)

                if do_save_ckpt:
                    checkpoint_path = os.path.join(
                        hparams.messenger.checkpoint_dir,
                        "checkpoint_epoch{}.pth".format(int(hparams.epoch)))
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        optimizer=optimizer,
                        save_optimizer_state=True,
                        model=model,
                        hparams=hparams,
                        test_loss_list=test_loss_list)
                    note_taking(
                        "Saving checkpoint to: {}".format(checkpoint_path))
                if do_logging:
                    sample_images(hparams, model, epoch)

    checkpoint_path = os.path.join(
        hparams.messenger.checkpoint_dir, "checkpoint_epoch{}.pth".format(
            int(hparams.epoch)))
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        optimizer=optimizer,
        save_optimizer_state=True,
        model=model,
        hparams=hparams,
        test_loss_list=test_loss_list)

    hparams.checkpoint_path = get_chechpoint_path(hparams)
    if hparams.checkpoint_path is None:
        hparams.chkt_step = -1
        note_taking(
            "Training done. Did not find the best checkpoint, loading from the latest instead: {}"
            .format(hparams.checkpoint_path))
        hparams.checkpoint_path = get_chechpoint_path(hparams)
    else:
        note_taking(
            "Training done. About to load checkpoint to eval from: {}".format(
                hparams.checkpoint_path))

    hparams.step, hparams.epoch, _ = load_checkpoint(
        path=hparams.checkpoint_path,
        optimizer=None,
        reset_optimizer=False,
        model=model)
    sample_images(hparams, model, hparams.epoch, best=True)

    return model
