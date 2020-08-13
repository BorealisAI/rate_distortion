# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang and Yanshuai Cao

#!/usr/bin/env python3

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import gc
import re
import sys
from lite_tracer import LTParser
from ..models.gans import *
from .experiment_utils import init_dir, note_taking
from torchvision.utils import save_image

import shutil


def get_args(args=None, parse_known=False):
    parser = LTParser()  #argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='./data/', help='path to dataset')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument(
        '--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--imageSize',
        type=int,
        default=32,
        help='the height / width of the input image to network')
    parser.add_argument('--loss_type', type=str, default='standard', help='')

    parser.add_argument('--g_type', type=str, default='dcgan', help='')

    parser.add_argument(
        '--nz', type=int, default=100, help='size of the latent z vector')

    parser.add_argument(
        '--ngf',
        type=int,
        default=128,
        help='channel number for conv and hidden size for mlp')
    parser.add_argument('--ndf', type=int, default=128)

    parser.add_argument('--ng_conv', type=int, default=4)
    parser.add_argument('--nd_conv', type=int, default=4)

    parser.add_argument('--d_sn', default=False, action='store_true')
    parser.add_argument('--g_sn', default=False, action='store_true')

    parser.add_argument('--spectral_d', type=int, default=0)
    parser.add_argument('--spectral_g', type=int, default=0)

    parser.add_argument('--d_nm', type=str, default='bn')
    parser.add_argument('--g_nm', type=str, default='bn')

    parser.add_argument('--gp_weight', type=float, default=0.)
    parser.add_argument('--bre_weight', type=float, default=0.)

    parser.add_argument(
        '--ng_update',
        type=int,
        default=200000,
        help='number of G update to train for')
    parser.add_argument(
        '--n_dstep', type=int, default=1, help='number of D update step')
    parser.add_argument(
        '--lr_d',
        type=float,
        default=0.0002,
        help='learning rate for D, default=0.0002')
    parser.add_argument(
        '--lr_g',
        type=float,
        default=0.0002,
        help='learning rate for G, default=0.0002')

    parser.add_argument(
        '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help='beta2 for adam. default=0.9')

    parser.add_argument(
        '--cuda', default=True, action='store_true', help='enables cuda')
    parser.add_argument(
        '--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument(
        '--netG', default='', help="path to netG (to continue training)")
    parser.add_argument(
        '--netD', default='', help="path to netD (to continue training)")
    parser.add_argument(
        '--outf',
        default='./results/',
        help='folder to output images and model checkpoints')
    parser.add_argument(
        '--manualSeed', type=int, default=42, help='manual seed')
    parser.add_argument('--namestr', type=str, help='note')

    if parse_known:
        opt, unknown = parser.parse_known_args(args)
        print('Unknown args:', unknown)
    else:
        opt = parser.parse_args(args)
        opt.outname_base = '{}_{}_{}'.format(opt.dataset, opt.namestr,
                                             opt.hash_code)
        opt.result_folder = os.path.join(opt.outf, opt.outname_base)
        opt.result_path = os.path.join(opt.result_folder, opt.hash_code)

    print(opt)
    return opt


def pm1_2_z1(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def to_z1(x):
    if x.min() >= 0 and x.max() <= 1.:
        return x
    elif x.min() < 0 and x.min() >= -1. and x.max() <= 1.:
        return pm1_2_z1(x)
    else:
        raise ValueError()


def best_tile_shape(n):
    n_sqrt = int(np.sqrt(np.float(n)))
    width = n // n_sqrt

    while (n // width) * width != n:
        width -= 1
    return int(width), int(n // width)


def backup_checkpoints(model, hparams):
    """ Backup the checkpoints and then exit. """
    init_dir(hparams.output_root_dir + "/checkpoints/gans/results/")

    model_path = hparams.specific_model_path.strip()
    hash_code = re.search('(?P<hash>LT_.*_LT)',
                          os.path.basename(model_path)).group('hash')
    ########################################################
    # what's the config of this model
    proj_dir = os.path.abspath(model_path).split('results/')[0]
    source_txt_path = os.path.join(proj_dir, 'lt_records', hash_code,
                                   'settings_' + hash_code + '.txt')
    try:
        backup_gan_path = hparams.output_root_dir + "/checkpoints/gans/results/" + hparams.specific_model_path[
            32:]
        backup_gan_dir = hparams.output_root_dir + "/checkpoints/gans/results/" + hparams.specific_model_path[
            32:].split("/")[0]
        init_dir(backup_gan_dir)
        torch.save(model.state_dict(), backup_gan_path)

    except:
        backup_gan_path = hparams.output_root_dir + "/checkpoints/gans/results/" + hparams.specific_model_path[
            56:]
        backup_gan_dir = hparams.output_root_dir + "/checkpoints/gans/results/" + hparams.specific_model_path[
            56:].split("/")[0]
        init_dir(backup_gan_dir)
        torch.save(model.state_dict(), backup_gan_path)

    target_model_path = backup_gan_path.strip()

    ########################################################
    # what's the config of this model
    target_proj_dir = os.path.abspath(target_model_path).split('results/')[0]
    target_txt_path = os.path.join(target_proj_dir, 'lt_records', hash_code,
                                   'settings_' + hash_code + '.txt')

    target_dir = os.path.join(target_proj_dir, 'lt_records', hash_code)
    init_dir(target_dir)
    dest = shutil.copyfile(source_txt_path, target_txt_path)

    note_taking("Copied text from {} to {}.".format(source_txt_path, dest))
    sys.exit()


def gan_bridge(hparams):
    ########################################################
    # what model
    N_SAMPLE = 1024
    model_path = hparams.specific_model_path.strip()
    hash_code = re.search('(?P<hash>LT_.*_LT)',
                          os.path.basename(model_path)).group('hash')
    ########################################################
    # what's the config of this model
    proj_dir = os.path.abspath(model_path).split('results/')[0]
    with open(
            os.path.join(proj_dir, 'lt_records', hash_code,
                         'settings_' + hash_code + '.txt'), 'r') as fr:

        args = fr.read().strip().split()

        args.insert(args.index('--netG') + 1, model_path)
        args.insert(args.index('--netD') + 1, '')

        opt = get_args(args, parse_known=True)
    ########################################################
    # configure and load this model
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    if opt.dataset == 'mnist':
        nc = 1
        img_dims = (1, opt.imageSize, opt.imageSize)
    elif opt.dataset == 'cifar10':
        nc = 3
        img_dims = (nc, opt.imageSize, opt.imageSize)
    else:
        raise NotImplementedError()

    if opt.g_type == 'dcgan':
        netG = DCGANGenerator(
            opt.ngf,
            opt.imageSize,
            opt.nz,
            nc,
            opt.ngpu,
            nm=opt.g_nm,
            spectral_norm=opt.spectral_g,
            n_convs=opt.ng_conv).to(device)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))

    elif opt.g_type == 'sheldon_mlp':

        netG = sheldonMLPGenerator(
            hdim=opt.ngf,
            img_dims=img_dims,
            zdim=opt.nz,
            spectral_norm=opt.g_sn,
            flat_out=False).to(device)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))

    elif opt.g_type == 'sheldon_mlp2':

        netG = sheldonMLPGenerator2(
            hdim=opt.ngf,
            img_dims=img_dims,
            zdim=opt.nz,
            spectral_norm=opt.g_sn,
            flat_out=False).to(device)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))

    elif opt.g_type == 'sheldon_mlp3':

        netG = sheldonMLPGenerator3(
            hdim=opt.ngf,
            img_dims=img_dims,
            zdim=opt.nz,
            spectral_norm=opt.g_sn,
            flat_out=False).to(device)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))

    else:
        raise NotImplementedError()

    print(netG)
    G_total_params = sum(
        p.numel() for p in netG.parameters() if p.requires_grad)
    print('G tot params: {:g}'.format(G_total_params))

    if hparams.backup_gans:
        backup_checkpoints(netG, hparams)
    ########################################################
    # adding variance, processing decoder for RD
    # netG = add_variance(netG, hparams)
    netG.x_logvar = torch.nn.Parameter(
        torch.log(torch.tensor(hparams.model_train.x_var)))

    hparams.model_train.z_size = opt.nz
    ########################################################
    # Sample from this model
    all_gen_images = []
    while len(all_gen_images) <= N_SAMPLE // opt.batchSize:
        with torch.no_grad():
            noise = torch.randn(opt.batchSize, opt.nz, device=device)
            fake_images = netG(noise)

            all_gen_images.append(to_z1(fake_images))
            gc.collect()

    all_gen_images = torch.cat(all_gen_images, 0)
    w, h = best_tile_shape(len(all_gen_images))
    gan_image_dir = hparams.messenger.image_dir + 'samples.png'
    save_image(all_gen_images, gan_image_dir, nrow=w)

    return netG
