import argparse

import train


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# import torchvision
# import torchvision.transforms as transforms
# from ... import data_generator as gn
# import data_generator_pytorch as gn
# import datetime
# import time

import datetime
import argparse
import logging
import os
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
# import model.data_loader as data_loader
import model.data_generator as data_generator
from evaluate import evaluate

from tensorboardX import SummaryWriter 


# set up the parser as in train.py 

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', default=32,
                    help="Size of immune embedding (default:8)")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="File containing directory containing datasets")
# parser.add_argument('--data_dir_list', default=None,
# help="File contating list of dataset directories data_dirs")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--prefix', default='',
                    help="Prefix of dataset files  \n \
                    (e.g. prefix=\"tcga\" implies input files are \n \
                    tcga_ssgsea_[train,test,val].txt, \n \
                    tcga_phenotype_[train,test,val].txt )")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--logging_dir', default=None, help="Optional, where you want to log Tensorboard too")

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')

parser.add_argument(
    '--smoke-test', action="store_true", help="Finish quickly for testing")

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




if __name__ == "__main__":
    args = parser.parse_args()

    import numpy as np
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

    ray.init()
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="c_index",
        max_t=400,
        grace_period=20)
    tune.register_trainable("train_deepimmune",
                            lambda cfg, rprtr: train.setup_and_train(args, cfg, rprtr))
    tune.run_experiments(
        {
            "exp": {
                "stop": {
                    "c_index": 0.98,
                    "training_iteration": 1 if args.smoke_test else 20
                },
                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": int(not args.no_cuda)
                },
                "run": "train_mnist",
                "num_samples": 1 if args.smoke_test else 10,
                "config": {
                    "lr": tune.sample_from(
                        lambda spec: np.random.uniform(0.001, 0.1))
                }
            }
        },
        verbose=0,
        scheduler=sched)