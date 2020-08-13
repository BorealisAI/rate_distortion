# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import os
from ..hparams.hparam import Hparam as container
import torch
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from torchvision import datasets, transforms
from datetime import datetime
import random
import logging as python_log
from tensorboardX import SummaryWriter
import copy
from torchvision.utils import save_image
from ..hparams.registry import get_hparams
from ..models.registry import get_model


class AISTracker(object):
    """
    Can be used to track individual chains in an AIS run and plot them separately. 
    This is not needed for all the plots in paper. 
    It is useful when you want investigate each independent chain. 
    Support only batch size of one when investigating each independent chain separately.
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.distortion_chains = list()
        self.log_weights_chains = list()
        self.log_Z = list()
        self.rate_chains = list()
        self.beta_array = None

    def track_distortion(self, distortion):
        self.distortion_chains.append(distortion)

    def track_logZ(self, logZ):
        self.log_Z.append(logZ)

    def track_logw(self, logw):
        self.log_weights_chains.append(copy.deepcopy(logw))

    def compute_rate_chains(self):
        self.beta_array = torch.unsqueeze(
            torch.from_numpy(self.hparams.messenger.beta_list),
            dim=1).to(self.hparams.tensor_type).to(self.hparams.device)
        self.log_weights_chains = torch.stack(self.log_weights_chains, dim=0)
        self.log_Z = torch.stack(self.log_Z, dim=0)
        self.distortion_chains = torch.stack(self.distortion_chains, dim=0)
        self.beta_array = self.beta_array.repeat(
            1, self.hparams.rd.num_total_chains)
        self.rate_chains = -self.log_weights_chains - (
            self.distortion_chains * self.beta_array)

    def plot_rd_chains(self):

        self.compute_rate_chains()
        for chain_index in range(self.hparams.rd.n_chains):
            distortions = self.distortion_chains[:, chain_index]
            rates = self.rate_chains[:, chain_index]
            label = "chain_{}".format(chain_index)
            plot_rate_distrotion(
                self.hparams,
                self.hparams.messenger.beta_list,
                rates.cpu().numpy(),
                distortions.cpu().numpy(),
                "test",
                metric=label)

    def plot_with_analytical(self, analytic_rate_list,
                             analytic_distortion_list):
        self.compute_rate_chains()
        for chain_index in range(self.hparams.rd.n_chains):
            distortions = self.distortion_chains[:, chain_index]
            rates = self.rate_chains[:, chain_index]
            label = "chain_{}".format(chain_index)
            plot_both(self.hparams, self.hparams.messenger.beta_list,
                      analytic_rate_list, analytic_distortion_list,
                      rates.cpu().numpy(),
                      distortions.cpu().numpy(), label)
            save_path = self.hparams.messenger.results_dir + '/' + self.hparams.hparam_set + '_csvlog_' + label
            header_csv = ',,,,,,,, b_s , n_chains , steps\n,,,,,,,, {} , {} , {} , \n'.format(
                self.hparams.rd.batch_size, self.hparams.rd.n_chains,
                self.hparams.messenger.total_anneal_steps)
            header_data = r"$\lambda$, D(ais), R(ais),D(analytical), R(analytical), logw, logZ"
            header_csv += header_data
            # need to revise for log_Z if batch size is greater than 1
            csv_array = np.stack(
                (self.beta_array[:, 0].cpu().numpy(), distortions.cpu().numpy(),
                 rates.cpu().numpy(), analytic_distortion_list,
                 analytic_rate_list,
                 self.log_weights_chains[:, chain_index].cpu().numpy(),
                 self.log_Z.squeeze().cpu().numpy()),
                axis=-1)
            np.savetxt(
                save_path + '.csv',
                csv_array,
                fmt='%.4f',
                delimiter=",",
                newline="\n",
                header=header_csv,
                comments='')


def note_taking(message, print_=True):
    if print_:
        print(message)
    python_log.info(message)


def compute_duration(start, end):
    duration = (end - start).total_seconds()
    days, _dreminder = divmod(duration, 86400)
    hours, _hreminder = divmod(_dreminder, 3600)
    minutes, seconds = divmod(_hreminder, 60)
    note_taking(
        "It's been {} days, {} hours, {} minutes and {} seconds.".format(
            days, hours, minutes, seconds))
    hour_summery = 24 * days + hours + minutes / 60.
    return hour_summery


def set_random_seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)


def compute_data_stats(loader, data):
    data_list = list()
    for i, (data, _) in enumerate(loader):
        data_list.append(data)
    whole_data = torch.cat(data_list, dim=0)
    note_taking("size of whole data: {}".format(whole_data.size()))

    mean = torch.mean(whole_data, dim=0)
    note_taking("size of mean: {}".format(mean.size()))
    std = torch.std(whole_data, dim=0)
    data_max = torch.max(whole_data)
    data_min = torch.min(whole_data)
    return mean, std, data_max, data_min


def infer_lambda_range(lower, upper, steps, decimal, extrsteps=None):
    """ 
    Infer the lambda range based on the lower and upper bound of lambda.
    Always include 1 for sanity check, unless 1 in not between lower and upper.  
    """
    if extrsteps is not None:
        lambda_range_small = np.linspace(lower, 1 if upper > 1 else upper,
                                         extrsteps) if lower < 1 else list()
    else:
        lambda_range_small = np.linspace(lower, 1 if upper > 1 else upper,
                                         steps) if lower < 1 else list()

    lambda_range_big = np.linspace(1 if lower < 1 else lower, upper,
                                   steps) if upper > 1 else list()

    lambda_range = np.append(
        lambda_range_small if len(lambda_range_small) == 0 else
        lambda_range_small[:-1], lambda_range_big
        if len(lambda_range_big) > 0 else [lambda_range_small[-1]])
    return lambda_range


def tempreture_scheduler_beta(max_beta, temperatures, hparams):
    """ 
    Given a tempreture schedule and a the maximum beta value,
    return a dictionary of tempreture schedules where 
    each beta as the key and corresponding tempreture schedule as the value.

    """

    full_temperatures = temperatures * max_beta
    num_betas = hparams.rd.num_betas
    num_temp = len(full_temperatures)
    start_indexs = np.linspace(
        0, num_temp - np.floor(num_temp / num_betas), num_betas, dtype=int)
    end_indexs = np.append(start_indexs[1:], [len(full_temperatures) - 1])

    if end_indexs[-1] == start_indexs[-1]:
        end_indexs = end_indexs[:-1]
        start_indexs = start_indexs[:-1]

    beta_list = full_temperatures[end_indexs]
    index_pairs = np.stack((start_indexs, end_indexs))
    tempreture_dict = {}
    total_anneal_steps = 0
    for i, current_beta in enumerate(beta_list, 0):
        (start_index, end_index) = index_pairs[:, i]
        current_tempretures = copy.deepcopy(
            full_temperatures[start_index:end_index + 1])
        tempreture_dict.update({current_beta: current_tempretures})
        total_anneal_steps += (len(current_tempretures) - 1)

    total_anneal_steps += 1
    hparams.messenger.total_anneal_steps = total_anneal_steps
    hparams.messenger.beta_list = beta_list
    note_taking(
        "Generated clean tempreture schedule with total anneal steps={}, for betas:{}"
        .format(total_anneal_steps, beta_list))

    return tempreture_dict


def extract_Wb(model, hparams):
    """ 
    Extract the weights and bias for analytical solution.
    """
    decoder_weights = model.dec_mean.weight
    decoder_bias = model.dec_mean.bias
    return decoder_weights, decoder_bias


def print_hparams(hparams_dict, name):
    """Prints the values of all hyper parameters.
    """
    if name is None:
        print("Sanity check: hyper parameters:")
    print('=' * 80)
    print('Hparams'.center(80) if name is None else str(name).center(80))
    print('-' * 80)
    for key, value in hparams_dict.items():
        if isinstance(value, dict):
            print_hparams(value, key)
        else:
            if "msg" in key:
                print('=' * 80)
                print(key.center(80))

                print('-' * 80)
                print(value)

            else:
                print('{}: {}'.format(key, value).center(80))
    print('*' * 80)


def log_hparams(file, hparams_dict, name):
    """Log down the values of all hyper parameters.
    """
    file.write('=' * 80)
    file.write("\n")
    file.write('Hparams'.center(80) if name is None else str(name).center(80))
    file.write("\n")
    file.write('-' * 80)
    file.write("\n")
    for key, value in hparams_dict.items():
        file.write("\n")
        if isinstance(value, dict):
            log_hparams(file, value, key)
        else:
            file.write('{}: {}'.format(key, value).center(80))
    file.write("\n")
    file.write('=' * 80)
    file.write("\n")


def logging(args_dict, hparams_dict, log_dir, dir_path, stage="init"):
    if stage == "init":
        log_path = log_dir + "init_log.txt"
    else:
        log_path = log_dir + str(stage) + "_log.txt"

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(log_path, 'a') as file:
        file.write("The current time when logging  is:" + current_time + "\n")
        file.write("The Current directory is:" + str(dir_path))
        file.write("\n Args for this experiment are:\n")
        for key, value in args_dict.items():
            file.write('%s:%s\n' % (key, value))
        file.write("\n")
        file.write("\nlogging hparam and results recursively...:\n")
        log_hparams(file, hparams_dict, args_dict["hparam_set"])
        file.close()


def init_dir(dir):
    if not os.path.exists(dir):
        print("Initializing directory: {}".format(dir))
        os.makedirs(dir, 0o777)
        os.chmod(dir, 0o777)


def init_rd_ais_results(data):
    rate_list = list()
    rd_const_lower_list = list()
    rd_const_dict = {"lower": rd_const_lower_list}

    if data == "simulate":
        rd_const_upper_list = list()
        rd_const_mean_list = list()
        rd_const_dict.update({"mean": rd_const_mean_list})
        rd_const_dict.update({"upper": rd_const_upper_list})

    distortion_list = list()
    L2_D_list = list()
    results = {
        "rate_list": rate_list,
        "rd_const_dict": rd_const_dict,
        "distortion_list": distortion_list,
        "L2_D_list": L2_D_list
    }
    return results


def sample_images(hparams,
                  model,
                  epoch,
                  best=False,
                  prior_dist=None,
                  name=None,
                  sample_z=None):
    if sample_z is None:
        sample = torch.randn(64, hparams.model_train.z_size).to(
            device=hparams.device, dtype=hparams.tensor_type)
    else:
        sample = sample_z
    sample, x_logvar = model.decode(sample)
    sample = sample.cpu()
    save_name = hparams.messenger.image_dir + (
        'best_' if best else
        '') + 'sample' + (str(epoch) if epoch is not None else '') + (
            ("_" + name) if name is not None else '')
    image_path = save_name + '.png'

    if hparams.dataset.data_name == "cifar10":
        sample = sample / 2 + 0.5
        note_taking("...cifar generation re-normalized and saved. ")
    save_image(
        sample.view(64, hparams.dataset.input_dims[0],
                    hparams.dataset.input_dims[1],
                    hparams.dataset.input_dims[2]), image_path)
    if x_logvar is not None:
        note_taking(
            "Image sampled from the {}checkpoint has the decoder variance: {}".
            format(("best " if best else ''),
                   torch.exp(x_logvar).detach().cpu().numpy().item()))
    else:
        note_taking("Image sampled from the {}checkpoint".format(
            ("best " if best else '')))


def load_user_model(hparams):
    model = get_model(hparams).to(hparams.device)
    load_user_checkpoint(path=hparams.checkpoint_path, model=model)
    return model


def build_BDMC_schedule(tempreture_dict, beta_list):
    schedule = tempreture_dict[beta_list[0]]
    if len(beta_list) > 1:
        for beta in beta_list[1:]:
            schedule = np.concatenate((schedule, tempreture_dict[beta][1:]))
    return schedule


def initialze_BDMC(tempreture_dict, hparams):
    """ 
    hparams.BDMC_points indicates how many points do we want, i.e, how many betas. 
    Then the beta values are taken linearly from the list of betas of the previous experiment.
    """
    assert hparams.BDMC_points is not None, "BDMC_points can not be {}".format(
        hparams.BDMC_points)
    beta_list = list(tempreture_dict.keys())
    note_taking("beta_list: {}".format(beta_list), print_=False)
    if hparams.beta_indexes is None:
        select_indexs = np.linspace(
            0, len(beta_list) - 1, hparams.BDMC_points, dtype=int)
    else:
        select_indexs = hparams.beta_indexes
    note_taking("select_indexs: {}".format(select_indexs), print_=False)
    BDMC_tempreture_dict = dict()

    for (i, index) in enumerate(select_indexs, 0):
        BDMC_tempreture_dict[beta_list[index]] = build_BDMC_schedule(
            tempreture_dict, beta_list[:(index + 1)])
    BDMC_betas = list(BDMC_tempreture_dict.keys())
    hparams.messenger.BDMC_betas = BDMC_betas
    note_taking("About to run BDMC on betas: {}".format(BDMC_betas))
    note_taking(
        "BDMC_tempreture_dict: {}".format(BDMC_tempreture_dict), print_=False)
    return BDMC_tempreture_dict, BDMC_betas, select_indexs


def initialze_run(hparams, args):
    """ 
    The helper function to initialize experiment. Mostly just book keeping.
    Naming convention: everything ending with "dir" ends with "/"
    Directory naming logic: 
        - If train_first, experiment will first run training algorithm, and then 
            run RD evaluation, 
        - If not in train_first mode, only RD evaluation will be run. 
        - If additional experiment name(e_name) is specified, then append that at the end
            this additional name is useful when one wishes to conduct multiple runs under the same hparam
            - Note taht e_name better only be used for testing purposes. It's a good practice to always 
                create a new hparam for a new experiment, even just chaning the random seed, 
                which can be done with simply having a same root hparam and then initialize branches. 
        Args:
        hparams: The hyper parameter object
        args: dictionary of keyword arguments. 

    """

    results_dir = hparams.output_root_dir + "/results_out/"
    backup_dir = hparams.output_root_dir + "/result_backup/"
    init_dir(backup_dir)
    if hparams.group_list:
        for subgroup in hparams.group_list:
            results_dir += (subgroup + "/")

    results_dir = results_dir + args.hparam_set + (
        "_" + str(args.e_name) if args.e_name is not None else "") + "/"
    init_dir(results_dir)
    hparams.save(results_dir + 'init_hparams.json')
    hparams.save(backup_dir + args.hparam_set + '_hparams.json')

    log_path = results_dir + "RT_log.txt"
    python_log.basicConfig(
        filename=log_path,
        filemode='a',
        level=python_log.INFO,
        format='%(message)s')
    arxiv_dir = results_dir + "result_arxiv/"
    image_dir = results_dir + "result_image/"

    init_dir(arxiv_dir)
    init_dir(image_dir)

    tboard_dir = hparams.output_root_dir + "/tboard/"
    if hparams.group_list:
        for subgroup in hparams.group_list:
            tboard_dir += (subgroup + "/")
    init_dir(tboard_dir)
    tboard_path = tboard_dir + (args.hparam_set + (
        "_" + str(args.e_name) if args.e_name is not None else ""))

    writer = SummaryWriter(tboard_path)
    checkpoint_dir = hparams.checkpoint_root_dir + "/checkpoints/" + (
        (args.hparam_set +
         ("_" + str(args.e_name) if args.e_name is not None else "")) if
        (hparams.train_first or
         hparams.load_hparam_name is None) else hparams.load_hparam_name) + "/"
    if "fid" in hparams.rd.target_dist:
        checkpoint_dir_rep = hparams.checkpoint_root_dir + "/rep_checkpoints/" + (
            (args.hparam_set +
             ("_" + str(args.e_name) if args.e_name is not None else "")) if
            (hparams.rep_train_first or
             hparams.load_rep is None) else hparams.load_rep) + "/"
        init_dir(checkpoint_dir_rep)

    # cuda init
    hparams.cuda = hparams.cuda and torch.cuda.is_available()
    hparams.device = torch.device("cuda" if hparams.cuda else "cpu")
    if hparams.double_precision:
        hparams.add_hparam(
            "dtype",
            torch.cuda.DoubleTensor if hparams.cuda else torch.DoubleTensor)
        hparams.add_hparam("tensor_type", torch.float64)
    else:
        hparams.add_hparam(
            "dtype",
            torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor)
        hparams.add_hparam("tensor_type", torch.float32)
    if hparams.verbose:
        note_taking("torch.cuda.current_device() {}".format(
            torch.cuda.current_device()))
        note_taking("hparams.dtype {}".format(hparams.dtype))
        note_taking("hparams.tensor_type {}".format(hparams.tensor_type))
    hparams.model_train.add_hparam("dtype", hparams.dtype)
    hparams.model_train.add_hparam("tensor_type", hparams.tensor_type)
    torch.set_default_tensor_type(hparams.dtype)

    # init hparams
    hparams.rd.num_total_chains = hparams.rd.batch_size * hparams.rd.n_chains
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # hparams.step_sizes_target is supposed to be the "hparam_set" of the target exp.
    if hparams.step_sizes_target is None:
        step_sizes_dir = arxiv_dir + "step_sizes/"
        init_dir(step_sizes_dir)
    else:
        target_hparams = get_hparams(hparams.step_sizes_target)
        step_sizes_dir = target_hparams.output_root_dir + "/results_out/"
        if target_hparams.group_list:
            for subgroup in target_hparams.group_list:
                step_sizes_dir += (subgroup + "/")
        step_sizes_dir = step_sizes_dir + hparams.step_sizes_target + "/"
        step_sizes_dir += "result_arxiv/step_sizes/"
        note_taking("Will load step sizes from:{}".format(step_sizes_dir))
        # check random seed.
        # The random see should be different from the run where the step sizes are being loaded
        if target_hparams.random_seed == hparams.random_seed:
            note_taking(
                "WARNING!\n Random seed({}) is the same with the experinet with the target step sizes!!"
                .format(hparams.random_seed))

    # messenger is a Hparam object, it can be regarded as the messenger acting as
    # the global info buffer for miscellaneous bits of information
    hparams.messenger = container()
    hparams.messenger.results_dir = results_dir
    hparams.messenger.image_dir = image_dir
    hparams.messenger.arxiv_dir = arxiv_dir
    hparams.messenger.backup_dir = backup_dir
    hparams.messenger.step_sizes_dir = step_sizes_dir
    hparams.messenger.checkpoint_dir = checkpoint_dir
    hparams.messenger.result_dict = dict()
    hparams.messenger.dir_path = dir_path
    hparams.messenger.log_path = log_path
    hparams.messenger.tboard_path = tboard_path
    hparams.messenger.save_data = True
    hparams.messenger.rd_betas_done = list()
    if "fid" in hparams.rd.target_dist:
        hparams.messenger.checkpoint_dir_rep = checkpoint_dir_rep

    # initial logging
    hparams.hparam_set = args.hparam_set
    hparams.e_name = args.e_name
    logging(vars(args), hparams.to_dict(), results_dir, dir_path)

    if hparams.train_first:
        if not os.path.exists(checkpoint_dir):
            hparams.resume_training = False
            os.makedirs(checkpoint_dir, 0o777)
            os.chmod(checkpoint_dir, 0o777)
            note_taking(
                "....initialized checkpoint_dir: {}".format(checkpoint_dir))
        if hparams.resume_training:
            hparams.checkpoint_path = get_chechpoint_path(
                hparams
            ) if hparams.specific_model_path is None else hparams.specific_model_path
    else:
        hparams.checkpoint_path = get_chechpoint_path(
            hparams
        ) if hparams.specific_model_path is None else hparams.specific_model_path

    if hparams.verbose:
        print_hparams(hparams.to_dict(), None)

    return writer


def log_down_likelihood(analytic_rate, analytic_distortion, data, hparams):
    ELBO = -analytic_rate - analytic_distortion
    hparams.messenger.result_dict.update({"ELBO_analytical_" + data: ELBO})
    note_taking("Analytical log-likelihood on {} data is: {}".format(
        data, ELBO))


def save_comparison(data,
                    recon_batch,
                    batch_size,
                    hparams,
                    beta=None,
                    path=None):
    if hparams.dataset.data_name == "cifar10":
        recon_batch = recon_batch / 2 + 0.5
    if hparams.messenger.save_data:
        if hparams.dataset.data_name == "cifar10":
            data = data / 2 + 0.5
        original_path = hparams.messenger.image_dir + "original_AIS.png"
        save_image(
            data.view(-1, hparams.dataset.input_dims[0],
                      hparams.dataset.input_dims[1],
                      hparams.dataset.input_dims[2])[:64],
            original_path,
            nrow=8)
        hparams.messenger.save_data = False
        test_data_npy_dir = hparams.messenger.arxiv_dir + "test_data/"
        init_dir(test_data_npy_dir)
        test_data_npy_path = test_data_npy_dir + "test_data.npz"
        label = hparams.dataset.label if hparams.dataset.label is not None else "mixed"
        data = data.cpu().numpy()
        np.savez(test_data_npy_path, data, label)

    if beta is not None:
        image_path = hparams.messenger.image_dir + "beta{}.png".format(beta)
        save_image(
            recon_batch.view(-1, hparams.dataset.input_dims[0],
                             hparams.dataset.input_dims[1],
                             hparams.dataset.input_dims[2])[:64],
            image_path,
            nrow=8)
    else:
        n = min(data.size(0), 8)
        comp_data = data.view(-1, hparams.dataset.input_dims[0],
                              hparams.dataset.input_dims[1],
                              hparams.dataset.input_dims[2])[:n]
        comp_rec = recon_batch.view(-1, hparams.dataset.input_dims[0],
                                    hparams.dataset.input_dims[1],
                                    hparams.dataset.input_dims[2])[:n]
        comparison = torch.cat([comp_data, comp_rec])
        image_path = path
        save_image(comparison.cpu(), image_path, nrow=n)


def plot_rate_distrotion(hparams,
                         betas,
                         rate_list,
                         distortion_list,
                         data,
                         metric="L2"):

    plt.figure()
    plt.plot(distortion_list, rate_list, 'r--', label="rate_list")
    plt.plot(distortion_list, rate_list, 'b.')

    leg = plt.legend(
        loc='upper right', ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    title = 'RD-AIS'
    plt.title(title, fontsize=12)
    plt.xlabel('distortion', fontsize=12)
    plt.ylabel('rate', fontsize=12, horizontalalignment='right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale("log")
    save_path = hparams.messenger.results_dir + '/' + hparams.hparam_set + '_rd_ais_' + metric + (
        ('_on_' + data) if data is not None else '')
    npz_path = hparams.messenger.arxiv_dir + '/' + hparams.hparam_set + '_rd_ais_' + metric + (
        ('_on_' + data) if data is not None else '')
    np.savez(npz_path + '.npz', betas, rate_list, distortion_list)
    backup_path = hparams.messenger.backup_dir + hparams.hparam_set + '_rd_ais_' + metric + (
        ('_on_' + data) if data is not None else '')
    np.savez(backup_path + '.npz', betas, rate_list, distortion_list)
    plt.savefig(save_path + '.pdf')
    plt.savefig(backup_path + '.pdf')
    plt.close()


def plot_analytic_rate_distrotion(hparams, betas, analytic_rate_list,
                                  analytic_distortion_list, data):

    plt.figure()
    plt.plot(
        analytic_distortion_list, analytic_rate_list, 'r', label="analytical")
    plt.plot(analytic_distortion_list, analytic_rate_list, 'b.')
    leg = plt.legend(
        loc='upper right', ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.title('RD-analytical', fontsize=12)
    plt.xlabel('distortion', fontsize=12)
    plt.ylabel('rate', fontsize=12, horizontalalignment='right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    save_path = hparams.messenger.results_dir + '/' + hparams.hparam_set + '_rd_alt_on_' + data
    npz_path = hparams.messenger.arxiv_dir + '/' + hparams.hparam_set + '_rd_alt_on_' + data
    np.savez(npz_path + '.npz', betas, analytic_rate_list,
             analytic_distortion_list)

    plt.savefig(save_path + '.pdf')
    plt.close()


def plot_both(hparams, betas, analytic_rate_list, analytic_distortion_list,
              ais_rate_list, distortion_list, data):
    """ 
    Plot both AIS RD curve and Analytical RD curve. 
     """

    if hparams.distortion_limit is not None:
        cut_idx = (np.abs(np.array(distortion_list) -
                          hparams.distortion_limit)).argmin()
        note_taking("AIS Distortion limit={}, at beta={} at index {}".format(
            hparams.distortion_limit, betas[cut_idx], cut_idx))
        distortion_list = distortion_list[cut_idx:]
        ais_rate_list = ais_rate_list[cut_idx:]
        betas = betas[cut_idx:]

        analytic_distortion_list = analytic_distortion_list[cut_idx:]
        analytic_rate_list = analytic_rate_list[cut_idx:]

    plt.figure()
    plt.plot(
        analytic_distortion_list, analytic_rate_list, 'r:', label="Analytical")
    plt.plot(distortion_list, ais_rate_list, 'b:', label="AIS")
    leg = plt.legend(
        loc='upper right', ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    # plt.title('AIS vs. Analytical Solution', fontsize=24)
    plt.xlabel('Distortion(Negative Log Likelihood)', fontsize=18)
    plt.ylabel('Rate', fontsize=18, horizontalalignment='right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale("log")
    save_path = hparams.messenger.results_dir + '/' + hparams.hparam_set + '_ais_analytical'
    backup_path = hparams.messenger.backup_dir + hparams.hparam_set + '_ais_analytical'
    plt.grid(b=True, axis="both")
    plt.savefig(save_path + '.pdf')
    plt.savefig(backup_path + '.pdf')
    plt.close()


def plot_both_baseline(hparams, analytic_rate_list, analytic_distortion_list,
                       ais_rate_list, distortion_list, baseline_rate,
                       baseline_distortion, data):
    """ 
    Plot both AIS RD curve and Analytical RD curve. 
     """
    rate_list = ais_rate_list
    plt.figure()
    plt.plot(
        analytic_distortion_list, analytic_rate_list, 'r:', label="Analytical")
    plt.plot(distortion_list, rate_list, 'b:', label="AIS")
    plt.plot(baseline_distortion, baseline_rate, 'g--', label="Baseline")
    leg = plt.legend(
        loc='upper right', ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('AIS vs. Analytical Solution', fontsize=24)
    plt.xlabel('Distortion(Negative Log Likelihood)', fontsize=18)
    plt.ylabel('Rate', fontsize=18, horizontalalignment='right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    save_path = hparams.messenger.results_dir + '/' + hparams.hparam_set + '_rd_both_on_' + data
    plt.grid(b=True, axis="both")
    plt.savefig(save_path + '.pdf')
    backup_path = hparams.messenger.backup_dir + hparams.hparam_set + '_rd_both_on_' + (
        ('_on_' + data) if data is not None else '')
    plt.savefig(backup_path + '.pdf')
    plt.close()


def get_chechpoint_path(hparams):
    """
    generate appropriate checkpointing path based on checkpoitn step. 
    """
    chkt_step = int(hparams.chkt_step)
    max_step = 0
    best_ckpt = None
    latest_ckpt = None
    for f in glob.glob(os.path.join(hparams.messenger.checkpoint_dir, "*.pth")):
        if "best" in f:
            best_ckpt = f
        else:
            x = int(f.split(".")[-2].split("epoch")[-1])
            if x > max_step:
                max_step = x
                latest_ckpt = f

    if chkt_step == -2:
        checkpoint_path = best_ckpt
        note_taking("the best Checkpoint detected: {}".format(checkpoint_path).
                    center(80))
    elif chkt_step == -1:
        checkpoint_path = latest_ckpt
        note_taking("the latest Checkpoint detected: {}".format(
            checkpoint_path).center(80))

    note_taking(
        "about to load checkpoint from: {}".format(checkpoint_path).center(80))
    return checkpoint_path


def load_checkpoint(
        path,
        optimizer,
        reset_optimizer,
        model,
):
    '''
    Load model from checkpoint with optimizer loading support
    '''
    note_taking("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    step = checkpoint["global_step"]
    epoch = checkpoint["global_epoch"]
    test_loss_list = checkpoint["test_loss_list"]
    if optimizer is not None:
        if not reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                note_taking("Loading optimizer state from {}".format(path))
                optimizer.load_state_dict(checkpoint["optimizer"])
    try:
        x_var = torch.exp(model.x_logvar).detach().cpu().numpy().item()

    except:
        note_taking(
            "WARNING! Something went wrong with decoder variance, please check. Overwriting to x_var=1.0"
        )
        x_var = 1.0
    note_taking(
        "Loaded checkpoint from {} at epoch {}(Step {}), and the test loss was {},decoder variance is {})"
        .format(path, epoch, step, test_loss_list[-1], x_var))
    return step, epoch, test_loss_list


def load_user_checkpoint(path, model):
    '''
    Load model checkpoint
    '''
    note_taking("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    note_taking("Loaded checkpoint from {}.".format(path))


def save_checkpoint(checkpoint_path, optimizer, save_optimizer_state, model,
                    hparams, test_loss_list):
    '''
    Save the pytorch model with optimizer saving support
    '''
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": hparams.step,
        "global_epoch": hparams.epoch,
        "test_loss_list": test_loss_list
    }, checkpoint_path)
