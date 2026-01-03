import os
import sys
import argparse
import yaml
import random
import time
import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from detectors import DETECTOR
from trainer.trainer import Trainer
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR


# ---------------------- argparse ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--detector_path', type=str, required=True)
parser.add_argument('--train_dataset', nargs="+")
parser.add_argument('--test_dataset', nargs="+")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ddp', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)


# ---------------------- utils ----------------------
def init_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_dataloader(config, mode):
    dataset = DeepfakeAbstractBaseDataset(config=config, mode=mode)

    if mode == 'train' and config['ddp']:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = (mode == 'train')

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['train_batchSize'] if mode == 'train' else config['test_batchSize'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config['workers'],
        collate_fn=dataset.collate_fn,
        drop_last=(mode == 'test' and config['test_dataset'] == 'DeepFakeDetection')
    )


def build_optimizer(model, config):
    opt = config['optimizer']
    if opt['type'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=opt['adam']['lr'],
            betas=(opt['adam']['beta1'], opt['adam']['beta2']),
            weight_decay=opt['adam']['weight_decay'],
            eps=opt['adam']['eps']
        )
    if opt['type'] == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=opt['sgd']['lr'],
            momentum=opt['sgd']['momentum'],
            weight_decay=opt['sgd']['weight_decay']
        )
    if opt['type'] == 'sam':
        return SAM(model.parameters(), optim.SGD,
                   lr=opt['sam']['lr'],
                   momentum=opt['sam']['momentum'])
    raise NotImplementedError


def build_scheduler(optimizer, config):
    if config['lr_scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, config['lr_step'], config['lr_gamma']
        )
    if config['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config['lr_T_max'], config['lr_eta_min']
        )
    if config['lr_scheduler'] == 'linear':
        return LinearDecayLR(optimizer, config['nEpochs'], config['nEpochs'] // 4)
    return None


# ---------------------- main ----------------------
def main():
    # load config
    with open(args.detector_path) as f:
        config = yaml.safe_load(f)
    with open('training/config/train_config.yaml') as f:
        config.update(yaml.safe_load(f))

    config['ddp'] = args.ddp
    config['local_rank'] = args.local_rank
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset

    seed = args.seed or random.randint(1, 10000)
    config['manualSeed'] = seed
    init_seed(seed, config['cuda'])

    logger = create_logger('training.log')

    if config['ddp']:
        dist.init_process_group(backend='nccl')
        logger.addFilter(RankFilter(0))

    logger.info(f"Seed: {seed}")
    logger.info(f"Model: {config['model_name']}")

    # data
    train_loader = build_dataloader(config, 'train')
    test_loaders = {
        name: build_dataloader({**config, 'test_dataset': name}, 'test')
        for name in config['test_dataset']
    }

    # model
    model = DETECTOR[config['model_name']](config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    metric = config['metric_scoring']

    trainer = Trainer(config, model, optimizer, scheduler, logger, metric)

    # train
    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        best_metric = trainer.train_epoch(
            epoch, train_loader, test_loaders
        )
        if best_metric is not None:
            logger.info(
                f"Epoch[{epoch}] {metric}: {parse_metric_for_print(best_metric)}"
            )

    logger.info(f"Training finished, best metric = {best_metric}")


if __name__ == '__main__':
    main()
