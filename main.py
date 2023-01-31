#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm

from utils.readdata import load_data
from dataloader import PreTrainDataset_IRT, PreTrainDataset_TRT_IRI, PretrainDataIterator, TestforPreTrainDataset
from dataloader import PreTrainDataset_inter, TestforPreTrainInterDataset 
from dataloader import TrainDataset, TestDataset, DataLoaderIterator
from utils import parser
from model import Model

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

class Log:
    def __init__(self, args, file_name):
        self.logger = logging.getLogger(file_name)
        self.logger.setLevel(logging.DEBUG)
        mode = 'a'
        log_path = args.save_path
        logfile = os.path.join(log_path, file_name)
        fmt = "%(asctime)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt)
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        if args.print_on_screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.logger.addHandler(fh)
        if args.print_on_screen:
            self.logger.addHandler(sh)

    def info(self, message):
        self.logger.info(message)

def save_model(model, optimizer, save_variable_list, args, name):
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        name+'_optimizer': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint_'+name)
    )

def log_metrics(mode, step, metrics, logger):
    for metric in metrics:
        logger.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    set_global_seed(args.seed)

    cur_time = parse_time()
    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join('./logs', args.dataset, cur_time)

    print("logging to", args.save_path)

    if not args.do_pretrain:
        pretrain_writer = SummaryWriter(logdir='./logs-debug/unused-tb', comment='pretrain')
    else:
        pretrain_writer = SummaryWriter(logdir=args.save_path, comment='pretrain')
    
    if not args.do_pretrain_inter:
        pretrain_inter_writer = SummaryWriter(logdir='./logs-debug/unused-tb', comment='pretrain_inter')
    else:
        pretrain_inter_writer = SummaryWriter(logdir=args.save_path, comment='pretrain_inter')
    
    if not args.do_train:
        train_writer = SummaryWriter(logdir='./logs-debug/unused-tb', comment='train')
    else:
        train_writer = SummaryWriter(logdir=args.save_path, comment='train')

    print('Loading data ...')
    train_user_item_pair, test_user_item_pair, train_user_set, test_user_set, test_inter_mat, item_tag, triplets_IRT, triplets_TRT, triplets_IRI, n_params= load_data(args)
    for n_name, n_value in n_params.items():
        print('%s: %d' % (n_name, n_value), end='.   ')
    print('')
    
    setting_logger = Log(args, 'setting')
    setting_logger.info('-------------------------------'*3)
    setting_logger.info('Dataset: %s' % args.dataset)
    setting_logger.info('user: %s' % n_params['n_users'])
    setting_logger.info('item: %s' % n_params['n_items'])
    setting_logger.info('tag: %s' % n_params['n_tags'])
    setting_logger.info('entity: %s' % n_params['n_entities'])
    setting_logger.info('relation: %s' % n_params['n_relations'])

    pretrain_max_step = (len(triplets_IRI) + len(triplets_IRT) + len(triplets_TRT)) * args.pretrain_epoch // args.pretrain_batch_size
    pretrain_warmup_step = pretrain_max_step//2

    pretrain_inter_max_step = len(item_tag) * args.pretrain_inter_epoch // args.pretrain_inter_batch_size
    pretrain_inter_warmup_step = pretrain_inter_max_step//2

    train_max_step = len(train_user_item_pair) * args.train_epoch // args.train_batch_size
    train_warmup_step = train_max_step//2

    print('Initing dataloader ...')
    
    random.shuffle(triplets_IRT)
    triplets_IRT_train = triplets_IRT
    triplets_IRT_test = triplets_IRT[int(len(triplets_IRT)*0.9):]
    if args.do_pretrain:
        pretrain_logger = Log(args, 'pretrain')
        pretrain_IRT_dataloader_neg_item = DataLoader(
            PreTrainDataset_IRT(triplets_IRT, triplets_IRT_train, args.pretrain_negative_sample_size, n_params, 'IRT-item'),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True,
            collate_fn=PreTrainDataset_IRT.collate_fn
        )
        pretrain_IRT_dataloader_neg_tag = DataLoader(
            PreTrainDataset_IRT(triplets_IRT, triplets_IRT_train, args.pretrain_negative_sample_size, n_params, 'IRT-tag'),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True,
            collate_fn=PreTrainDataset_IRT.collate_fn
        )   
    
        pretrain_TRT_dataloader = DataLoader(
            PreTrainDataset_TRT_IRI(triplets_TRT, args.pretrain_negative_sample_size, n_params, 'TRT'),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True,
            collate_fn=PreTrainDataset_TRT_IRI.collate_fn
        )
        pretrain_IRI_dataloader = DataLoader(
            PreTrainDataset_TRT_IRI(triplets_IRI, args.pretrain_negative_sample_size, n_params, 'IRI'),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True,
            collate_fn=PreTrainDataset_TRT_IRI.collate_fn
        )

        IRT_len = len(triplets_IRT)
        TRT_len = len(triplets_TRT)
        IRI_len = len(triplets_IRI)
        IRT_ratio = IRT_len / (IRT_len+TRT_len+IRI_len)
        TRT_ratio = TRT_len / (IRT_len+TRT_len+IRI_len)
        IRI_ratio = IRI_len / (IRT_len+TRT_len+IRI_len)
        pretrain_dataloader = PretrainDataIterator(
            pretrain_IRT_dataloader_neg_item, pretrain_IRT_dataloader_neg_tag,
            pretrain_TRT_dataloader, pretrain_IRI_dataloader,
            IRT_ratio, TRT_ratio, IRI_ratio
        )
        pretrain_logger.info('pretrain_IRT: %d' % IRT_len)
        pretrain_logger.info('pretrain_TRT: %d' % TRT_len)
        pretrain_logger.info('pretrain_IRI: %d' % IRI_len)
        pretrain_logger.info('pretrain_IRT_ratio: %f' % IRT_ratio)
        pretrain_logger.info('pretrain_TRT_ratio: %f' % TRT_ratio)
        pretrain_logger.info('pretrain_IRI_ratio: %f' % IRI_ratio)
    if args.pretrain_do_test:
        pretrain_test_logger = Log(args, 'pretrain_test')
        test_pretrain_dataloader = DataLoader(
            TestforPreTrainDataset(triplets_IRT, triplets_IRT_test, n_params, 'IRT-tag', args),
            batch_size=args.pretrain_test_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True,
            collate_fn=TestforPreTrainDataset.collate_fn
        )
    
    data = []
    for item, rel_tags in item_tag.items():
        data.append([item, rel_tags])
    data_test = data[int(len(data)*0.95): ]
    data_train = data


    if args.do_pretrain_inter:
        pretrain_inter_logger = Log(args, 'pretrain_inter')
        pretrain_inter_dataloader = DataLoader(
            PreTrainDataset_inter(item_tag, data_train, args.pretrain_inter_negative_sample_size, n_params),
            batch_size=args.pretrain_inter_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True
        )
        pretrain_inter_dataloader = DataLoaderIterator(pretrain_inter_dataloader)
        pretrain_inter_logger.info('pretrain_inter: %d' % len(item_tag))
    if args.pretrain_inter_do_test:
        pretrain_inter_test_logger = Log(args, 'pretrain_inter_test')
        test_pretrain_inter_dataloader = DataLoader(
            TestforPreTrainInterDataset(item_tag, data_test, n_params, args),
            batch_size=args.pretrain_inter_test_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True,
        )
   

    if args.do_train:
        train_logger = Log(args, 'train')
        train_dataloader = DataLoader(
            TrainDataset(train_user_item_pair, train_user_set, item_tag, args, n_params),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True
        )
        train_dataloader = DataLoaderIterator(train_dataloader)
        train_logger.info('train: %d' % len(train_user_item_pair))

        
    if args.train_do_test:
        test_logger = Log(args, 'test')
        test_dataloader = DataLoader(
            TestDataset(train_user_set, test_user_set, item_tag, n_params),
            batch_size=args.train_test_batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            pin_memory=True
        )
        test_logger.info('test: %d' % len(test_user_item_pair))

    print('Initing model ...')

    I2B = Model(args, n_params)
    
    setting_logger.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in I2B.named_parameters():
        setting_logger.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    setting_logger.info('Parameter Number: %d' % num_params)
    
    if args.cuda:
        I2B = I2B.cuda()
    
    pretrain_lr = args.pretrain_learning_rate
    pretrain_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, I2B.parameters()), 
        lr=pretrain_lr
    )
    pretrain_inter_lr = args.pretrain_inter_learning_rate
    pretrain_inter_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, I2B.parameters()), 
        lr=pretrain_inter_lr
    )
    train_lr = args.train_learning_rate
    train_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, I2B.parameters()), 
        lr=train_lr
    )

    pretrain_step = 0
    pretrain_inter_step = 0
    train_step = 0 

    if args.checkpoint_path is not None:

        print('Loading model %s' % args.checkpoint)
        setting_logger.info('Loading checkpoint %s...' % args.checkpoint_path + '/checkpoint_' + args.checkpoint)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint))
        I2B.load_state_dict(checkpoint['model_state_dict'])
        
        if args.checkpoint == 'checkpoint_pretrain':
            pretrain_step = checkpoint['pretrain_step']
            pretrain_lr = checkpoint['pretrain_lr']
            pretrain_warmup_step = checkpoint['pretrain_warmup_step']
            pretrain_optimizer.load_state_dict(checkpoint['pretrain_optimizer'])
        elif args.checkpoint == 'checkpoint_pretrain_inter':
            pretrain_inter_step = checkpoint['pretrain_inter_step']
            pretrain_inter_lr = checkpoint['pretrain_inter_lr']
            pretrain_inter_warmup_step = checkpoint['pretrain_inter_warmup_step']
            pretrain_inter_optimizer.load_state_dict(checkpoint['pretrain_inter_optimizer'])
        else:
            train_step = checkpoint['train_step']
            train_lr = checkpoint['train_lr'] 
            train_warmup_step = checkpoint['train_warmup_step']
            train_optimizer.load_state_dict(checkpoint['train_optimizer'])

    else:
        setting_logger.info('Ramdomly Initializing Model...')

   
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    if args.do_pretrain:
        print('Doing pretrain ...')
        pretrain_logger.info('Start pretraining ...')
        pretrain_logger.info('Max epoch: %s' % args.pretrain_epoch)
        pretrain_logger.info('Max step: %s' % pretrain_max_step)
        pretrain_logger.info('Current step: %s' % pretrain_step)
        pretrain_logger.info('Warm up step: %s' % pretrain_warmup_step)
        pretrain_logger.info('Batch size: %s' % args.pretrain_batch_size)
        pretrain_logger.info('Learning rate: %s' % pretrain_lr)
        
        training_logs = []

        for step in range(pretrain_step, pretrain_max_step):
            log = I2B.train_step(I2B, pretrain_optimizer, pretrain_dataloader, args, 'pretrain')
            for metric in log:
                pretrain_writer.add_scalar(metric, log[metric], step)
            training_logs.append(log)

            if step >= pretrain_warmup_step:
                pretrain_lr = pretrain_lr / 3
                pretrain_logger.info('Change learning_rate to %f at step %d' % (pretrain_lr, step))
                pretrain_optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, I2B.parameters()),
                    lr=pretrain_lr
                )
                pretrain_warmup_step = pretrain_warmup_step * 1.5

            if step % args.log_step == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('PreTraining average', step, metrics, pretrain_logger)
                training_logs = []
            
            pretrain_step = step

        save_variable_list = {
                'pretrain_step': pretrain_step,
                'pretrain_lr': pretrain_lr,
                'pretrain_warmup_step': pretrain_warmup_step
            }
        save_model(I2B, pretrain_optimizer, save_variable_list, args, 'pretrain')

    if args.do_pretrain_inter:
        print('Doing pretrain intersection ...')

        pretrain_inter_logger.info('Start pretraining_inter ...')
        pretrain_inter_logger.info('Max epoch: %s' % args.pretrain_inter_epoch)
        pretrain_inter_logger.info('Max step: %s' % pretrain_inter_max_step)
        pretrain_inter_logger.info('Current step: %s' % pretrain_inter_step)
        pretrain_inter_logger.info('Warm up step: %s' % pretrain_inter_warmup_step)
        pretrain_inter_logger.info('Batch size: %s' % args.pretrain_inter_batch_size)
        pretrain_inter_logger.info('Learning rate: %s' % pretrain_inter_lr)
        
        training_logs = []
        best_performance = 0.0

        for step in range(pretrain_inter_step, pretrain_inter_max_step):
            
            log = I2B.train_step(I2B, pretrain_inter_optimizer, pretrain_inter_dataloader, args, 'pretrain_inter')
            for metric in log:
                pretrain_inter_writer.add_scalar(metric, log[metric], step)
            training_logs.append(log)

            if step >= pretrain_inter_warmup_step:
                pretrain_inter_lr = pretrain_inter_lr / 5
                pretrain_inter_logger.info('Change learning_rate to %f at step %d' % (pretrain_inter_lr, step))
                pretrain_inter_optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, I2B.parameters()),
                    lr=pretrain_inter_lr
                )
                pretrain_inter_warmup_step = pretrain_inter_warmup_step * 1.5
            
            if step % args.log_step == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('PreTraining inter average', step, metrics, pretrain_inter_logger)
                training_logs = []
            
            pretrain_inter_step = step

        save_variable_list = {
                'pretrain_inter_step': pretrain_inter_step,
                'pretrain_inter_lr': pretrain_inter_lr,
                'pretrain_inter_warmup_step': pretrain_inter_warmup_step
            }
        save_model(I2B, pretrain_inter_optimizer, save_variable_list, args, 'pretrain_inter')
    
    if args.do_train:
        print('Doing train ...')

        train_logger.info('Start training ...')
        train_logger.info('Max epoch: %s' % args.train_epoch)
        train_logger.info('Max step: %s' % train_max_step)
        train_logger.info('Current step: %s' % train_step)
        train_logger.info('Warm up step: %s' % train_warmup_step)
        train_logger.info('Batch size: %s' % args.train_batch_size)
        train_logger.info('Learning rate: %s' % train_lr)
        
        training_logs = []
        best_performance = 0.0

        for step in range(train_step, train_max_step):
            
            log = I2B.train_step(I2B, train_optimizer, train_dataloader, args, 'train')
            for metric in log:
                train_writer.add_scalar(metric, log[metric], step)
            training_logs.append(log)

            if step >= train_warmup_step:
                train_lr = train_lr / 5
                train_logger.info('Change learning_rate to %f at step %d' % (train_lr, step))
                train_optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, I2B.parameters()),
                    lr=train_lr
                )
                train_warmup_step = train_warmup_step * 1.5

            if args.train_do_test:
                if step % (args.test_epoch * train_max_step // args.train_epoch)  == 0 and step > 0:
                    train_logger.info('Evaluating for train in %s step ...' % step)
                    metrics = I2B.test_step(I2B, test_dataloader, args, 'train', test_user_set)
                    log_metrics('train performance', step, metrics, train_logger)
                    save_variable_list = {
                        'train_step': step,
                        'train_lr': train_lr,
                        'train_warmup_step': train_warmup_step
                    }
                    performance = metrics['recall@20']
                    if performance >= best_performance:
                        best_performance = performance
                        save_model(I2B, train_optimizer, save_variable_list, args, 'train')
                    else:
                        train_logger.info('Skip saving model!')
            
            if step % args.log_step == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('training average', step, metrics, train_logger)
                training_logs = []
            
            train_step = step
        
        if args.train_do_test:
            train_logger.info('Evaluating for train in %s step ...' % train_step)
            metrics = I2B.test_step(I2B, test_dataloader, args, 'train', test_user_set)
            log_metrics('train performance', train_step, metrics, train_logger)
            save_variable_list = {
                'train_step': train_step,
                'train_lr': train_lr,
                'train_warmup_step': train_warmup_step
            }
            performance = metrics['recall@20']
            if performance >= best_performance:
                best_performance = performance
                save_model(I2B, train_optimizer, save_variable_list, args, 'train')
            else:
                train_logger.info('Skip saving model!')
    
    pretrain_writer.close()
    pretrain_inter_writer.close()
    train_writer.close()

if __name__ == '__main__':
    main(parser.parse_args())