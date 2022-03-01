import argparse
import json


def get_default_args():
    parser = argparse.ArgumentParser(description='DefaultArgs')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=100, metavar='N', help='number of epochs')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval_epoch', type=int, default=1, metavar='N', help='frequency of evaluation')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='N', help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, metavar='N', help='hidden dim')
    parser.add_argument('--model', type=str, default='EGNN', metavar='N', help='the model to employ')
    parser.add_argument('--n_layers', type=int, default=4, metavar='N', help='number of layers for the autoencoder')
    parser.add_argument('--dataset', type=str, default="QM9", metavar='N', help='the dataset')
    parser.add_argument('--weight_decay', type=float, default=1e-6, metavar='N', help='weight decay')
    parser.add_argument('--data_dir', type=str, default='saved_dataset', help='Data directory.')

    args = parser.parse_args()
    return args


def load_params(args, param_path):
    with open(param_path, 'r') as f:
        params = json.load(f)
    for k in params:
        setattr(args, k, params[k])
    return args

