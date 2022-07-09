import sys

sys.path.append('./')
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pysign.utils import get_default_args, load_params, set_seed
from pysign.utils.transforms import ToFullyConnected
from pysign.dataset import NBody
import torch_geometric.transforms as T
import torch

lw = 1.7
fig = plt.figure(figsize=(4, 4), dpi=100)
ax = fig.gca(projection='3d')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

idx = 30  # the idx of the system

param_path = 'examples/configs/nbody_dynamics_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)


class NBody_transform(object):
    def __call__(self, data):
        data.edge_attr = data.charge[data.edge_index[0]] * data.charge[data.edge_index[1]]
        data.x = torch.norm(data.v, dim=-1, keepdim=True)
        data['z'] = data.charge  # for TFN and SE3-Tr.
        return data


dataset = NBody(root=args.data_path, transform=T.Compose([ToFullyConnected(), NBody_transform()]),
                n_particle=args.n_particle, num_samples=args.num_samples, T=args.T, sample_freq=args.sample_freq,
                num_workers=20, initial_step=args.initial_step, pred_step=args.pred_step)
test_dataset = dataset[900: 1000]

with open('/apdcephfs/share_1364275/jiaqihan/pysign_exps/exps/nbodyeval/EGNN/eval_result.pkl', 'rb') as f:
    all_loss, all_pred = pkl.load(f)  # [tot_G, T, N, 3]
charge = test_dataset[idx].charge.numpy()
print(charge)
pred = all_pred[idx]


def animate(T):
    plt.cla()
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(-3.3, 2)
    ax.set_ylim(-2, 3)
    ax.set_zlim(-4, 4)
    color = ['red' if _ == 1 else 'blue' for _ in charge]
    ax.scatter(pred[T][..., 0], pred[T][..., 1], pred[T][..., 2], color=color, s=40)
    return fig,


anim = animation.FuncAnimation(fig, animate,
                               frames=pred.shape[0], interval=30, blit=True)

f = "animation_" + str(idx) + ".gif"
writergif = animation.PillowWriter(fps=10)
anim.save(f, writer=writergif)
print('write gif to', f)
