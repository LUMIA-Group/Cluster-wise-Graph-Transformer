import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GINConv

from N2C_Prop_L import Double_Level_MessageProp_random_walk_wo_norm, Double_Level_KeyProp_random_walk_wo_norm, Double_Level_MessageProp_random_walk_w_norm, Double_Level_KeyProp_random_walk_w_norm
from einops.layers.torch import Rearrange


def get_convs(args):

    convs = nn.ModuleList()

    _input_dim = args.num_features+args.pos_enc_rw_dim+args.pos_enc_lap_dim
    _output_dim = args.num_hidden

    for _ in range(args.num_convs):

        if args.conv == 'GCN':

            conv = GCNConv(_input_dim, _output_dim)

        elif args.conv == 'GIN':

            conv = GINConv(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ), train_eps=False)

        convs.append(conv)

        _input_dim = _output_dim
        _output_dim = _output_dim

    return convs


def get_input_transform(args):

    return nn.Sequential(
        nn.Linear(args.num_features+args.pos_enc_rw_dim +
                  args.pos_enc_lap_dim, args.num_hidden),
        nn.ReLU(),
        nn.Dropout(p=args.dropout),
        nn.Linear(args.num_hidden, args.num_hidden),
        nn.ReLU(),
        nn.Dropout(p=args.dropout)
    )


def get_classifier(args):

    if args.residual == 'cat':
        return nn.Sequential(
            nn.Linear(args.num_hidden*2, args.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden, args.num_classes)
        )
    else:
        return nn.Sequential(
            nn.Linear(args.num_hidden, args.num_hidden//2),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden//2, args.num_classes)
        )


def get_deepset_layer(args, input_dim, output_dim, num_layers):
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, args.num_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=args.dropout))
        for _ in range(1, num_layers-1):
            layers.append(nn.Linear(args.num_hidden, args.num_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=args.dropout))
        layers.append(nn.Linear(args.num_hidden, output_dim))
    if args.layernorm:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


class Cluster_GT(torch.nn.Module):

    def __init__(self, args):

        super(Cluster_GT, self).__init__()

        self.args = args
        self.use_rw = args.pos_enc_rw_dim > 0
        self.use_lap = args.pos_enc_lap_dim > 0
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.nhid = args.num_hidden
        self.attention_based_readout = args.attention_based_readout
        self.prop_w_norm_on_coarsened = args.prop_w_norm_on_coarsened
        self.residual = args.residual
        self.remain_k1 = args.remain_k1
        self.d_k_tensor = torch.tensor(args.num_hidden).float()

        self.kernel_method = args.kernel_method
        assert self.kernel_method in ['relu', 'elu']

        if args.use_gnn:
            self.convs = get_convs(args)
        else:
            self.input_transform = get_input_transform(args)

        if args.attention_based_readout:
            self.readout_seed_vector = nn.Parameter(
                torch.randn(args.num_hidden), requires_grad=True)

        self.classifier = get_classifier(args)

        self.subgraph_combined_pre_deepset = get_deepset_layer(
            args, args.num_hidden, args.num_hidden, args.deepset_layers)
        self.subgraph_combined_post_deepset = get_deepset_layer(
            args, args.num_hidden, args.num_hidden, args.deepset_layers)

        self.diffQ = args.diffQ
        if args.diffQ:
            self.subgraph_combined_pre_deepset_prime = get_deepset_layer(
                args, args.num_hidden, args.num_hidden, args.deepset_layers)
            self.subgraph_combined_post_deepset_prime = get_deepset_layer(
                args, args.num_hidden, args.num_hidden, args.deepset_layers)

        self.subgraph_combined_linK = nn.Linear(
            args.num_hidden, args.num_hidden)
        self.subgraph_combined_linV = nn.Linear(
            args.num_hidden, args.num_hidden)

        self.patch_rw_dim = args.pos_enc_patch_rw_dim
        if self.patch_rw_dim > 0:
            if self.residual == 'cat':
                self.patch_rw_encoder = nn.Linear(
                    self.patch_rw_dim, 2 * args.num_hidden)
            else:
                self.patch_rw_encoder = nn.Linear(
                    self.patch_rw_dim, args.num_hidden)

        if args.prop_w_norm_on_coarsened:
            self.propM = Double_Level_MessageProp_random_walk_wo_norm(
                node_dim=-3)
            self.propK = Double_Level_KeyProp_random_walk_wo_norm(node_dim=-2)
        else:
            self.propM = Double_Level_MessageProp_random_walk_w_norm(
                node_dim=-3)
            self.propK = Double_Level_KeyProp_random_walk_w_norm(node_dim=-2)

        self.reshape = Rearrange('(B p) d ->  B p d', p=args.n_patches)

        self.alpha_beta = nn.Parameter(
            torch.randn(size=(2,)), requires_grad=True)

        self.layernorm = args.layernorm
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.num_hidden)

    def forward(self, data):
        # node PE
        if not self.use_rw:
            x = data.x
        else:
            x = torch.cat((data.x, data.rw_pos_enc), dim=-1)
        if self.use_lap:
            x = torch.cat((x, data.lap_pos_enc), dim=-1)
        # input transform
        if self.args.use_gnn:
            for _ in range(self.args.num_convs):
                x = F.relu(self.convs[_](x, data.edge_index))
                if self.layernorm:
                    x = self.layernorm(x)
        else:
            x = self.input_transform(x)

        # get subgraph-level data
        subgraph_combined_x = x[data.subgraphs_nodes_mapper]
        subgraph_combined_batch = data.subgraphs_batch

        # get edges of coarsened graph
        subgraphs_batch_row = data.subgraphs_batch_row
        subgraphs_batch_col = data.subgraphs_batch_col
        coarsen_edge_attr = data.coarsen_edge_attr
        coarsen_edge_index = torch.stack(
            [data.subgraphs_batch_row, data.subgraphs_batch_col], dim=0)

        if not self.prop_w_norm_on_coarsened:
            # compute laplacian of coarsened graph
            coarsen_deg = torch.bincount(
                subgraphs_batch_row, coarsen_edge_attr)
            coarsen_deg_inv_sqrt = coarsen_deg.pow(-0.5)
            coarsen_deg_inv_sqrt[coarsen_deg_inv_sqrt == float('inf')] = 0
            coarsen_edge_attr = coarsen_deg_inv_sqrt[subgraphs_batch_row] * \
                coarsen_edge_attr * coarsen_deg_inv_sqrt[subgraphs_batch_col]

        # compute query
        subgraph_combined_Q = self.subgraph_combined_pre_deepset(
            subgraph_combined_x)
        scattered_subgraph_combined_Q = scatter(
            subgraph_combined_Q, subgraph_combined_batch, dim=0, reduce="sum")
        scattered_subgraph_combined_Q = self.subgraph_combined_post_deepset(
            scattered_subgraph_combined_Q)

        if self.diffQ:
            # compute query prime
            subgraph_combined_Q_prime = self.subgraph_combined_pre_deepset_prime(
                subgraph_combined_x)
            scattered_subgraph_combined_Q_prime = scatter(
                subgraph_combined_Q_prime, subgraph_combined_batch, dim=0, reduce="sum")
            scattered_subgraph_combined_Q_prime = self.subgraph_combined_post_deepset_prime(
                scattered_subgraph_combined_Q_prime)

        # compute key and value
        subgraph_combined_K = self.subgraph_combined_linK(subgraph_combined_x)
        scattered_subgraph_combined_K = scatter(
            subgraph_combined_K, subgraph_combined_batch, dim=0, reduce="mean")
        subgraph_combined_V = self.subgraph_combined_linV(subgraph_combined_x)

        # kernelized
        if self.kernel_method == 'relu':
            kernelized_scattered_subgraph_combined_Q = F.relu(
                scattered_subgraph_combined_Q)
            kernelized_subgraph_combined_K = F.relu(subgraph_combined_K)
            kernelized_scattered_subgraph_combined_K = F.relu(
                scattered_subgraph_combined_K)
            if self.diffQ:
                kernelized_scattered_subgraph_combined_Q_prime = F.relu(
                    scattered_subgraph_combined_Q_prime)
        elif self.kernel_method == 'elu':
            kernelized_scattered_subgraph_combined_Q = 1 + \
                F.elu(scattered_subgraph_combined_Q)
            kernelized_subgraph_combined_K = 1 + F.elu(subgraph_combined_K)
            kernelized_scattered_subgraph_combined_K = 1 + \
                F.elu(scattered_subgraph_combined_K)
            if self.diffQ:
                kernelized_scattered_subgraph_combined_Q_prime = 1 + \
                    F.elu(scattered_subgraph_combined_Q_prime)

        alpha_beta = F.softmax(self.alpha_beta)
        sqrt_alpha = alpha_beta[0].pow(-0.5)
        sqrt_beta = alpha_beta[1].pow(-0.5)

        # concate double-level keys and queries
        concated_key = torch.hstack(
            (sqrt_alpha*kernelized_scattered_subgraph_combined_K[subgraph_combined_batch], sqrt_beta*kernelized_subgraph_combined_K))
        if self.diffQ:
            concated_query = torch.hstack(
                (sqrt_alpha*kernelized_scattered_subgraph_combined_Q, sqrt_beta*kernelized_scattered_subgraph_combined_Q_prime))
        else:
            concated_query = torch.hstack(
                (sqrt_alpha*kernelized_scattered_subgraph_combined_Q, sqrt_beta*kernelized_scattered_subgraph_combined_Q))

        # scatter double-level keys
        scattered_concated_key = scatter(
            concated_key, subgraph_combined_batch, dim=0, reduce="sum")

        # compute message and scatter kernelized message
        kernelized_subgraph_combined_M = torch.einsum(
            'ni,nj->nij', [concated_key, subgraph_combined_V])
        scattered_kernelized_subgraph_combined_M = scatter(
            kernelized_subgraph_combined_M, subgraph_combined_batch, dim=0, reduce="sum")

        # propagate message and key on the coarsened graph
        if not self.prop_w_norm_on_coarsened:
            scattered_kernelized_subgraph_combined_M = self.propM(
                scattered_kernelized_subgraph_combined_M, coarsen_edge_index, coarsen_edge_attr.view(-1, 1, 1),)
            scattered_concated_key = self.propK(
                scattered_concated_key, coarsen_edge_index, coarsen_edge_attr.view(-1, 1), )
        else:
            scattered_kernelized_subgraph_combined_M = self.propM(
                scattered_kernelized_subgraph_combined_M, coarsen_edge_index,)
            scattered_concated_key = self.propK(
                scattered_concated_key, coarsen_edge_index, )

        # compute attention
        if self.diffQ:
            kernelized_subgraph_combined_H = torch.einsum(
                'ni,nij->nj', [concated_query, scattered_kernelized_subgraph_combined_M])
            kernelized_subgraph_combined_C = torch.einsum(
                'ni,ni->n', [concated_query, scattered_concated_key]).unsqueeze(-1) + 1e-6
        else:
            kernelized_subgraph_combined_H = torch.einsum(
                'ni,nij->nj', [concated_query, scattered_kernelized_subgraph_combined_M])
            kernelized_subgraph_combined_C = torch.einsum(
                'ni,ni->n', [concated_query, scattered_concated_key]).unsqueeze(-1) + 1e-6
        out = kernelized_subgraph_combined_H / kernelized_subgraph_combined_C

        # residual connection
        if self.residual in ['sum', 'cat']:
            scattered_subgraph_combined_x = scatter(
                subgraph_combined_x, subgraph_combined_batch, dim=0, reduce="mean")
            if self.residual == 'sum':
                out = out + scattered_subgraph_combined_x
            elif self.residual == 'cat':
                out = torch.cat((out, scattered_subgraph_combined_x), dim=-1)

        # Patch PE
        if self.patch_rw_dim > 0:
            out += self.patch_rw_encoder(data.patch_pe)

        # reshape from (number of patches of the whole batch, hidden_dim) to (graph_id, number of patches, hidden_dim)
        out = self.reshape(out)

        # attention-based readout
        if self.attention_based_readout:
            inner_products = torch.einsum(
                'ijk,k->ij', out, self.readout_seed_vector)
            readout_attention_weights = F.softmax(inner_products, dim=-1)
            out = torch.einsum('ij,ijk->ik', readout_attention_weights, out)
        # average pooling
        else:
            out = (out * data.mask.unsqueeze(-1)).sum(1) / \
                data.mask.sum(1, keepdim=True)

        # output decoder
        out = self.classifier(out)

        return F.log_softmax(out, dim=-1)
