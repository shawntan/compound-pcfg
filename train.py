#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from utils import *
from models import CompPCFG
from torch.nn.init import xavier_uniform_
from seq_dataloader import data_loader

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')

# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    src_path_train = 'data/simple/train_wo_valid_random.src'
    trg_path_train = 'data/simple/train_wo_valid_random.trg'
    src_path_valid = 'data/simple/valid.random.src'
    trg_path_valid = 'data/simple/valid.random.trg'

    src_vocab = set(w.lower() for l in open(src_path_train)
            for w in l.strip().split())
    trg_vocab = set(w.lower() for l in open(trg_path_train)
            for w in l.strip().split())
    src_vocab.update(['<pad>', '<start>', '<end>', '<unk>'])
    src_id2w = list(src_vocab)
    src_id2w.sort()
    src_w2id = {w: i for i, w in enumerate(src_id2w)}
    trg_vocab.update(['<pad>', '<start>', '<end>', '<unk>'])
    trg_id2w = list(trg_vocab)
    trg_id2w.sort()
    trg_w2id = {w: i for i, w in enumerate(trg_id2w)}

    train_data = data_loader.get_loader(
            src_path_train, trg_path_train,
            src_w2id, trg_w2id,
            batch_size=1
            )

    val_data = data_loader.get_loader(
            src_path_valid, trg_path_valid,
            src_w2id, trg_w2id,
            batch_size=1
            )

    # val_data = Dataset(args.val_file)
    # train_sents = train_data.batch_size.sum()
    in_vocab_size = len(src_id2w)
    out_vocab_size = len(trg_id2w)
    # max_len = max(val_data.sents.size(1), train_data.sents.size(1))
    # print('Train: %d sents / %d batches, Val: %d sents / %d batches' %
    #       (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
    # print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
    print('Save Path', args.save_path)
    cuda.set_device(args.gpu)
    model = CompPCFG(
            in_vocab = in_vocab_size,
            out_vocab = out_vocab_size,
            state_dim = args.state_dim,
            t_states = args.t_states,
            nt_states = args.nt_states,
            h_dim = args.h_dim,
            w_dim = args.w_dim,
            z_dim = args.z_dim
            )
    for name, param in model.named_parameters():    
        if param.dim() > 1:
            xavier_uniform_(param)
    print("model architecture")
    print(model)
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
    best_val_ppl = 1e5
    best_val_f1 = 0
    epoch = 0
    while epoch < args.num_epochs:
        start_time = time.time()
        epoch += 1
        print('Starting epoch %d' % epoch)
        train_nll = 0.
        train_kl = 0.
        num_sents = 0.
        num_words = 0.
        all_stats = [[0., 0., 0.]]
        b = 0
        for data in train_data:
            b += 1
            inp, inp_len, trg, trg_len = data
            inp = inp.to('cuda' if args.cuda else 'cpu').permute(1, 0)
            trg = trg.to('cuda' if args.cuda else 'cpu')
            nll, kl = model(inp, trg)
            print(nll)
            (nll+kl).mean().backward()
            train_nll += nll.sum().item()
            train_kl += kl.sum().item()

            if b % 16 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)      
                optimizer.step()
                optimizer.zero_grad()


            if b % args.print_every == 0:
                all_f1 = get_f1(all_stats)
                param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
                gparam_norm = sum([p.grad.norm()**2 for p in model.parameters() 
                    if p.grad is not None]).item()**0.5
                log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                        ', ValPPL: %.2f, ValF1: %.2f, ' + \
                        'CorpusF1: %.2f, Throughput: %.2f examples/sec'
                print(log_str %
                        (epoch, b, len(train_data), param_norm, gparam_norm, args.lr, 
                            best_val_ppl, best_val_f1, 
                            all_f1[0], num_sents / (time.time() - start_time)))


        args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
        print('--------------------------------')
        print('Checking validation perf...')    
        val_score = eval(val_data, model)
        print('--------------------------------')
#        if val_ppl < best_val_ppl:
#          best_val_ppl = val_ppl
#          best_val_f1 = val_f1
#          checkpoint = {
#            'args': args.__dict__,
#            'model': model.cpu(),
#            'word2idx': train_data.word2idx,
#            'idx2word': train_data.idx2word
#          }
#          print('Saving checkpoint to %s' % args.save_path)
#          torch.save(checkpoint, args.save_path)
#          model.cuda()

def eval(data_iter, model):
    model.eval()
    with torch.no_grad():
        for data in data_iter:
            inp, inp_len, trg, trg_len = data
            inp = inp.to('cuda' if args.cuda else 'cpu').permute(1, 0)
            trg = trg.to('cuda' if args.cuda else 'cpu')

            # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
            # but we don't for eval since we want a valid upper bound on PPL for early stopping
            # see eval.py for proper MAP inference
            output = model.sample(inp)
            print(output)
            print(' '.join(trg_id2w[idx] for idx in output))

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = True
    main(args)
