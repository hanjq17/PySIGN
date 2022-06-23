import argparse
import os
import time
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import sys

sys.path.append('.')
from airgeom.nn.model.atom3d_model import GNN_LBA
from airgeom.dataset.atom3d import GNNTransformLBA
from atom3d.datasets import LMDBDataset, PTGDataset
from scipy.stats import spearmanr
from airgeom.utils import load_params, set_seed


def train_loop(epoch, model, loader, optimizer, device):
    model.train()
    start = time.time()
    loss_all = 0
    total = 0
    print_frequency = 50
    for it, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(
                f'Epoch {epoch}, iter {it}, train loss {np.sqrt(loss.item())}, avg it/sec {print_frequency / elapsed}')
            start = time.time()

    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())

    r_p = np.corrcoef(y_true, y_pred)[0, 1]
    r_s = spearmanr(y_true, y_pred)[0]

    print(f'\tRMSE {np.sqrt(loss_all / total)}, Pearson {r_p}, Spearman {r_s}')

    return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)
    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_path, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_path, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_path, 'val'))
    else:
        transform = GNNTransformLBA()
        train_dataset = LMDBDataset(os.path.join(args.data_path, 'train'), transform=transform)
        val_dataset = LMDBDataset(os.path.join(args.data_path, 'val'), transform=transform)
        test_dataset = LMDBDataset(os.path.join(args.data_path, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break

    model = GNN_LBA(num_features, hidden_dim=args.hidden_dim).to(device)
    model.to(device)

    best_val_loss = 999
    best_rp = 0
    best_rs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epoch + 1):
        start = time.time()
        train_loss = train_loop(epoch, model, train_loader, optimizer, device)
        val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, device)
        if val_loss < best_val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(
            '\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss,
                                                                                                   r_p, r_s))

    if test_mode:
        train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(model, train_loader, device)
        torch.save({'targets': y_true_train, 'predictions': y_pred_train}, train_file)
        _, _, _, y_true_val, y_pred_val = test(model, val_loader, device)
        torch.save({'targets': y_true_val, 'predictions': y_pred_val}, val_file)
        rmse, pearson, spearman, y_true_test, y_pred_test = test(model, test_loader, device)
        print(f'\tTest RMSE {rmse}, Test Pearson {pearson}, Test Spearman {spearman}')
        torch.save({'targets': y_true_test, 'predictions': y_pred_test}, test_file)

    return best_val_loss, best_rp, best_rs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)  # 1e-5
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', type=bool, default=False)
    args = parser.parse_args()

    param_path = 'examples/configs/atom3d_lba.json'
    args = load_params(args, param_path=param_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.model_save_path
    set_seed(args.seed)

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)

    elif args.mode == 'test':
        seed = 99
        print('seed:', seed)
        assert log_dir is not None
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train(args, device, log_dir, rep=None, test_mode=True)

        # for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
        #     print('seed:', seed)
        #     log_dir = os.path.join('logs', f'lba_test_{args.seqid}')
        #     if not os.path.exists(log_dir):
        #         os.makedirs(log_dir)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     train(args, device, log_dir, rep, test_mode=True)
