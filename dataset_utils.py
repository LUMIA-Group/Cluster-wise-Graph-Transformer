import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

import os

from dataset_transform import PositionalEncodingTransform, GraphPartitionTransform


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(args, sparse=True, cleaned=False, normalize=False):
    name = args.data

    pe_transform = PositionalEncodingTransform(
        rw_dim=args.pos_enc_rw_dim, lap_dim=args.pos_enc_lap_dim)

    dataset = TUDataset(os.path.join('./data', name), name,
                        use_node_attr=True, cleaned=cleaned, pre_transform=pe_transform)
    dataset.data.edge_attr = None

    if args.n_patches > 0:
        subgraph_transform = GraphPartitionTransform(n_patches=args.n_patches,
                                                     is_directed=name == 'TreeDataset',
                                                     patch_rw_dim=args.pos_enc_patch_rw_dim,
                                                     patch_num_diff=args.pos_enc_patch_num_diff)

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.Compose(
                [T.OneHotDegree(max_degree), subgraph_transform])
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = T.Compose(
                [NormalizedDegree(mean, std), subgraph_transform])

    elif normalize:

        dataset.data.x -= torch.mean(dataset.data.x, axis=0)
        dataset.data.x /= torch.std(dataset.data.x, axis=0)

    if not sparse:
        max_num_nodes = 0
        for data in dataset:
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        if dataset.transform is None:
            dataset.transform = T.Compose(
                [T.ToDense(max_num_nodes), subgraph_transform])
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(max_num_nodes)])

    if dataset.transform is None:
        dataset.transform = subgraph_transform

    return dataset
