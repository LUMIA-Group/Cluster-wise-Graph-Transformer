import time
import argparse
import random
import time
import numpy as np
from tqdm import tqdm, trange
from functools import reduce

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup


from eval import eval

from data import load_data, num_graphs


# Parser - add_argument
parser = argparse.ArgumentParser(description='Cluster_GT')

# data setting
parser.add_argument('--data', default='IMDB-MULTI', type=str,
                    choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY',
                             'IMDB-MULTI', 'MUTAG', 'COLLAB', 'ENZYMES'],
                    help='dataset type')

# model setting
parser.add_argument("--model", type=str,
                    default='Cluster_GT', choices=['Cluster_GT'])
parser.add_argument("--model-string", type=str, default='Cluster_GT')

# fixed setting
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument("--grad-norm", type=float, default=1.0)
parser.add_argument("--lr-schedule", action='store_true')
parser.add_argument("--normalize", action='store_true')
parser.add_argument('--num-epochs', default=300,
                    type=int, help='train epochs number')
parser.add_argument('--patience', type=int, default=30,
                    help='patience for earlystopping')


# training setting
parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')
parser.add_argument('--batch-size', default=64,
                    type=int, help='train batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight-decay', type=float,
                    default=0.00001, help='weight decay')
parser.add_argument("--dropout", type=float, default=0.)

# gpu setting
parser.add_argument("--gpu", type=int, default=0)

# input transformation setting
parser.add_argument("--use-gnn", action='store_true')
parser.add_argument("--conv", type=str, default='GIN', choices=['GCN', 'GIN'])
parser.add_argument("--num-convs", type=int, default=1)

# hyper-parameter for model arch
parser.add_argument("--online", action='store_true')
parser.add_argument("--layernorm", action='store_true')
parser.add_argument("--remain-k1", action='store_true')
parser.add_argument("--diffQ", action='store_true')
parser.add_argument("--residual", type=str, default='cat',
                    choices=['None', 'cat', 'sum'])
parser.add_argument("--kernel_method", type=str,
                    default='elu', choices=['relu', 'elu'])
parser.add_argument("--deepset-layers", type=int, default=2)
parser.add_argument("--pos-enc-rw-dim", type=int, default=8)
parser.add_argument("--pos-enc-lap-dim", type=int, default=0)
parser.add_argument("--n-patches", type=int, default=8)
parser.add_argument("--prop-w-norm-on-coarsened", action='store_true')
parser.add_argument("--pos-enc-patch-rw-dim", type=int, default=0)
parser.add_argument("--pos-enc-patch-num-diff", type=int, default=-1)
parser.add_argument("--attention-based-readout", action='store_true')
parser.add_argument("--convex-linear-combination", action='store_true')


args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = args.gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

torch.set_num_threads(1)

dataset = load_data(args)

print(f"Dataset: {args.data}")
print(f"Input transfrom: {'GNN' if args.use_gnn else 'MLP'}")
print(f'Metis Online: {args.online}')
print(f"Model: {args.model}")
print(f"Device: {args.device}")

overall_results = {
    'best_val_loss': [],
    'best_val_acc': [],
    'best_test_loss': [],
    'best_test_acc': [],
    'durations': []
}

train_fold_iter = tqdm(range(1, 11), desc='Training')
val_fold_iter = [i for i in range(1, 11)]


for fold_number in train_fold_iter:
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

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False)

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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_epochs//10, num_training_steps=args.num_epochs)

    patience = 0
    best_loss = 1e9
    best_val_acc = 0
    best_test_loss = 1e9
    best_test_acc = 0

    t_start = time.perf_counter()

    for epoch in trange(0, (args.num_epochs), desc='[Epoch]', position=1):
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
            best_val_acc = val_acc
            best_val_epoch = epoch
            patience = 0

            best_test_acc = test_acc
            best_test_loss = test_loss
        else:
            patience += 1

        # print('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f%%' % (
        #     fold_number, epoch, total_loss, val_loss, val_acc))

        # print("[Val: Fold %d-Epoch %d] (Loss) Best Val Loss: %.2f Best Val Acc: %.2f%% at Epoch: %d" % (
        #     fold_number, epoch, best_loss, best_val_acc, best_val_epoch))

        train_fold_iter.set_description('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f TestAcc: %.2f' % (
            fold_number, epoch, total_loss, val_loss, val_acc, test_acc))
        train_fold_iter.refresh()

        if patience > args.patience:
            break

    t_end = time.perf_counter()

    overall_results['durations'].append(t_end - t_start)
    overall_results['best_val_loss'].append(best_loss)
    overall_results['best_val_acc'].append(best_val_acc)
    overall_results['best_test_loss'].append(best_test_loss)
    overall_results['best_test_acc'].append(best_test_acc)

    print("[Test: Fold {}] Test Acc: {} with Time: {}".format(
        fold_number, best_test_acc, (t_end - t_start)))

print("Overall result - overall_best_val: {} with std: {}; overall_best_test: {} with std: {}\n".format(
    np.array(overall_results['best_val_acc']).mean(),
    np.array(overall_results['best_val_acc']).std(),
    np.array(overall_results['best_test_acc']).mean(),
    np.array(overall_results['best_test_acc']).std()
))
