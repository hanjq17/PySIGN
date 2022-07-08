import sys
sys.path.append('./')
from pysign.nn.model import get_model_from_args
from pysign.task import Prediction
from pysign.utils import get_default_args, load_params, set_seed
from pysign.utils.transforms import ToFullyConnected
from torch_geometric.data import Data, Batch
import torch
import numpy as np
from scipy.linalg import qr

param_path = 'examples/configs/nbody_test_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

transform = ToFullyConnected()


def equivariance_test(model, decoding, vector_method):

    args.model = model
    rep_model = get_model_from_args(node_dim=5, edge_attr_dim=0, args=args, dynamics=True)
    task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim if model != 'RF' else 5, task_type='Regression', loss='MAE',
                    decoding=decoding, vector_method=vector_method, target='vector', dynamics=True, return_outputs=True)
    task.eval()
    data_list = []
    data_trans_list = []
    N = 10
    Q = np.random.randn(3, 3)
    Q = qr(Q)[0]
    Q = Q / np.linalg.det(Q)
    Q = torch.from_numpy(np.array(Q)).float()
    t = torch.from_numpy(np.random.randn(1, 3)).float()
    for i in range(10):
        x = torch.rand(N, 3)
        v = torch.rand(N, 3)
        h = torch.rand(N, 5)
        v_label = -v
        data = Data(x=x, v=v, h=h, v_label=v_label)
        data_trans = Data(x=x @ Q + t, v = v @ Q, h=h, v_label = v_label @ Q)
        data = transform(data)
        data_trans.edge_index = data.edge_index
        data_list.append(data)
        data_trans_list.append(data_trans)
    batch = Batch.from_data_list(data_list)
    batch_trans = Batch.from_data_list(data_trans_list)
    _,_, outputs = task(batch)
    output = outputs['vector'][0]
    _,_, outputs_trans = task(batch_trans)
    output_trans = outputs_trans['vector'][0]
    dis = torch.sum(torch.abs(output @ Q - output_trans))
    print('Roto-translation eq error:', dis.item())

def invariance_test(model):

    args.model = model
    rep_model = get_model_from_args(node_dim=5, edge_attr_dim=0, args=args, dynamics=True)
    task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='Regression', loss='MAE',
                        decoding='MLP', vector_method=None, scalar_pooling='sum', target='scalar', return_outputs=True)
    task.eval()
    data_list = []
    data_trans_list = []
    N = 10
    Q = np.random.randn(3, 3)
    Q = qr(Q)[0]
    Q = torch.from_numpy(np.array(Q)).float()
    t = torch.from_numpy(np.random.randn(1, 3)).float()
    for i in range(10):
        x = torch.rand(N, 3)
        v = torch.rand(N, 3)
        h = torch.rand(N, 5)
        y = torch.rand(1)
        data = Data(x=x, v=v, h=h, y=y)
        data_trans = Data(x=x @ Q + t, v = v @ Q, h=h, y=y)
        data = transform(data)
        data_trans.edge_index = data.edge_index
        data_list.append(data)
        data_trans_list.append(data_trans)
    batch = Batch.from_data_list(data_list)
    batch_trans = Batch.from_data_list(data_trans_list)
    _,_, outputs = task(batch)
    output = outputs['scalar'][0]
    _,_, outputs_trans = task(batch_trans)
    output_trans = outputs_trans['scalar'][0]
    dis = torch.sum(torch.abs(output - output_trans))
    print('Roto-translation eq error:', dis.item())

if __name__ == '__main__':

    model_map = {
        'TFN': [('MLP', 'diff')],
        'SE3Transformer': [('MLP', 'diff')],
        'SchNet': [('MLP', 'gradient')],
        'DimeNet': [('MLP', 'gradient')],
        'EGNN': [('MLP', 'diff'), ('MLP', 'gradient')],
        'RF': [('MLP', 'diff')],
        'PaiNN': [('MLP', 'gradient'), ('GatedBlock', None)],
        'ET': [('MLP', 'gradient'), ('GatedBlock', None)],
    }

    for model in model_map:

        for decoder in model_map[model]:
            print("="*5, f"Equivariance Test of {model} & {decoder}", "="*5)
            equivariance_test(model, *decoder)
        if model not in ['RF']:
            print("="*5, f"Invariance Test of {model}", "="*5)
            invariance_test(model)            