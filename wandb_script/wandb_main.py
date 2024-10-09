import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.realpath('.'))

import random
import numpy as np
import wandb
import json
import hashlib
import traceback
import time
from argparse import Namespace
from tqdm import tqdm, trange
from functools import reduce

from torch_geometric.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

from eval import eval
from data import load_data, num_graphs





def get_hash(dict_in, hash_keys, ignore_keys):
    dict_in = {k: v for k, v in dict_in.items() if k in hash_keys}
    dict_in = {k: v for k, v in dict_in.items() if k not in ignore_keys}
    hash_out = hashlib.blake2b(json.dumps(
        dict_in, sort_keys=True).encode(), digest_size=4).hexdigest()
    return str(hash_out)

def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runner(sweep_id, gpu_index, code_fullname, save_model):
    dir_name = 'remote'

    wandb.init(dir=dir_name, reinit=True, group=sweep_id)

    try:
        wandb.use_artifact(code_fullname, type='code')

        params_hash = get_hash(
            wandb.config, wandb.config['hash_keys'], wandb.config['ignore_keys'])
        wandb.config.update({'params_hash': params_hash},
                            allow_val_change=True)
        wandb.config.update({'gpu_index': gpu_index}, allow_val_change=True)

        args = Namespace(**dict(wandb.config))
        print("This trial's parameters: %s" % (args))

        fixSeed(args.seed)

        use_cuda = gpu_index >= 0 and torch.cuda.is_available()
        if use_cuda:
            # torch.cuda.set_device(gpu_index)
            args.device = 'cuda:{}'.format(gpu_index)
        else:
            args.device = 'cpu'

        print('Using device:', args.device)

        torch.set_num_threads(1)

        dataset = load_data(args)

        train_fold_iter = tqdm(range(1, 11), desc='Training')
        val_fold_iter = [i for i in range(1, 11)]


        fold_number = args.fold
        val_fold_number = val_fold_iter[fold_number - 2]

        train_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/train_idx-%d.txt' % (args.data, fold_number),
                                                dtype=np.int32), dtype=torch.long)
        val_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (args.data, val_fold_number),
                                                dtype=np.int32), dtype=torch.long)     
        test_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (args.data, fold_number),
                                                dtype=np.int32), dtype=torch.long)

        all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
        assert len(all_idxes) == len(dataset)

        train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))
        
        train_set, val_set, test_set = dataset[train_idxes], dataset[val_idxes], dataset[test_idxes]

        if not args.online:
            train_set = [x for x in train_set]
        val_set = [x for x in val_set]
        test_set = [x for x in test_set]
        
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

        if args.model == 'Cluster_GT':
            if args.convex_linear_combination:
                from Cluster_GT_N2C_L import Cluster_GT
            else:
                from Cluster_GT_N2C_T import Cluster_GT
            model = Cluster_GT(args)
        else:
            raise ValueError("Model Name <{}> is Unknown".format(args.model))
        
        if use_cuda:
            model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        if args.lr_schedule:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_epochs//10, num_training_steps=args.num_epochs)
        
        patience = 0
        best_loss = 1e9
        
        t_start = time.perf_counter()

        for epoch in trange(0, (args.num_epochs), desc = '[Epoch]', position = 1):
            model.train()
            total_loss = 0

            for _, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(args.device)
                out = model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                total_loss += loss.item() * num_graphs(data)
                optimizer.step()

                if args.lr_schedule:
                    scheduler.step()

            total_loss = total_loss / len(train_loader.dataset)

            val_acc, val_loss = eval(val_loader, model, args)
            test_acc, test_loss = eval(test_loader, model, args)
            
            if val_loss < best_loss:
                best_loss = val_loss         
                patience = 0
                best_test_acc = test_acc
            else:
                patience += 1


            # print('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f%%' % (
            #     fold_number, epoch, total_loss, val_loss, val_acc))

            # print("[Val: Fold %d-Epoch %d] (Loss) Best Val Loss: %.2f Best Val Acc: %.2f%% at Epoch: %d" % (
            #     fold_number, epoch, best_loss, best_val_acc, best_val_epoch))
            
            train_fold_iter.set_description('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f TestAcc: %.2f' % (
                fold_number, epoch, total_loss, val_loss, val_acc, test_acc))
            train_fold_iter.refresh()

            # wandb.log({'metric/val': val_acc, 'metric/test': test_acc, 'loss/train': total_loss, 'loss/val': val_loss, 'loss/test': test_loss})

            if patience > args.patience:
                break 

        
        t_end = time.perf_counter()
        wandb.run.summary["metric/final"] = best_test_acc
        wandb.run.summary["metric/time"] = t_end - t_start
        print("[Test: Fold {}] Test Acc: {} with Time: {}".format(fold_number, best_test_acc, (t_end - t_start)))






    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

    wandb.finish()
    return "success"