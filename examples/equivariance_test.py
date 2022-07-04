from models import SymNet
import torch
import numpy as np
from scipy.linalg import qr

np.random.seed(99)
torch.random.manual_seed(99)
gravity_axis = 1


def compute_rotation_matrix(theta, d):
    x, y, z = torch.unbind(d, dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    ret = torch.stack((
        cos + (1 - cos) * x * x,
        (1 - cos) * x * y - sin * z,
        (1 - cos) * x * z + sin * y,
        (1 - cos) * x * y + sin * z,
        cos + (1 - cos) * y * y,
        (1 - cos) * y * z - sin * x,
        (1 - cos) * x * z - sin * y,
        (1 - cos) * y * z + sin * x,
        cos + (1 - cos) * z * z,
    ), dim=-1)
    return ret.reshape(3, 3)


model = SymNet(n_layer=4, s_dim=5, hidden_dim=64, cutoff=10, gravity_axis=gravity_axis).cuda()
N = 100
x = torch.rand(N, 3).cuda()
v = torch.rand(N, 3).cuda()
h = torch.rand(N, 5).cuda()

M = 10
obj_id = torch.zeros(N // M)
obj_id = torch.cat([obj_id + _ for _ in range(M)]).cuda().long()
obj_type = ['rigid' for _ in range(M)]
obj_type[-1] = 'cloth'

v_out = model(x, v, h, obj_id, obj_type)
print(v_out.shape)
print(v_out[:5])
# temp = model.r

if gravity_axis is None:
    Q = np.random.randn(3, 3)
    Q = qr(Q)[0]
    Q = torch.from_numpy(np.array(Q)).float().cuda()
else:
    d = torch.zeros(3)
    d[gravity_axis] = 1
    Q = compute_rotation_matrix(torch.randn(1), d).cuda()

t = torch.from_numpy(np.random.randn(1, 3)).float().cuda()
print(t)

# Translation
_x = x + t
_v_out = model(_x, v, h, obj_id, obj_type)
# _temp = model.r
dis = torch.sum(torch.abs(v_out - _v_out))
print('Translation eq error:', dis.item())

# Rotation
_x = x @ Q
_v = v @ Q
_v_out = model(_x, _v, h, obj_id, obj_type)
dis = torch.sum(torch.abs(v_out @ Q - _v_out))
print('Rotation eq error:', dis.item())

# Rotation + Translation
_x = x @ Q + t
_v = v @ Q
_v_out = model(_x, _v, h, obj_id, obj_type)
dis = torch.sum(torch.abs(v_out @ Q - _v_out))
print('Roto-translation eq error:', dis.item())