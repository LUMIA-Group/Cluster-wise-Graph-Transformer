import numpy as np

from dataset_utils import get_dataset


def load_data(args):

    dataset = get_dataset(args, normalize=args.normalize)
    args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
        np.mean([data.num_nodes for data in dataset]))
    print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' %
          (dataset, args.num_features, args.num_classes, args.avg_num_nodes))

    return dataset


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
