import torch
import numpy as np


class torch_default_dtype:

    def __init__(self, dtype):
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.saved_dtype)


def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())
    return torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    '''
    ZYZ Eurler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.get_default_dtype())
    x = x / torch.norm(x)
    beta = torch.acos(x[2])
    alpha = torch.atan2(x[1], x[0])
    return (alpha, beta)


# These functions (x_to_alpha_beta and rot) satisfies that
# rot(*x_to_alpha_beta([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]


def irr_repr(order, alpha, beta, gamma, dtype=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    return torch.tensor(wigner_D_matrix(order, np.array(alpha), np.array(beta), np.array(gamma)),
                        dtype=torch.get_default_dtype() if dtype is None else dtype)


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = torch.atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def kron(x, y):
    assert x.ndimension() == 2
    assert y.ndimension() == 2
    return torch.einsum("ij,kl->ikjl", (x, y)).view(x.size(0) * y.size(0), x.size(1) * y.size(1))


################################################################################
# Change of basis
################################################################################


def xyz_vector_basis_to_spherical_basis():
    """
    to convert a vector [x, y, z] transforming with rot(a, b, c)
    into a vector transforming with irr_repr(1, a, b, c)
    see assert for usage
    """
    with torch_default_dtype(torch.float64):
        A = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)
        assert all(torch.allclose(irr_repr(1, a, b, c) @ A, A @ rot(a, b, c)) for a, b, c in torch.rand(10, 3))
    return A.type(torch.get_default_dtype())


def tensor3x3_repr(a, b, c):
    """
    representation of 3x3 tensors
    T --> R T R^t
    """
    r = rot(a, b, c)
    return kron(r, r)


def tensor3x3_repr_basis_to_spherical_basis():
    """
    to convert a 3x3 tensor transforming with tensor3x3_repr(a, b, c)
    into its 1 + 3 + 5 component transforming with irr_repr(0, a, b, c), irr_repr(1, a, b, c), irr_repr(3, a, b, c)
    see assert for usage
    """
    with torch_default_dtype(torch.float64):
        to1 = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(0, a, b, c) @ to1, to1 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

        to3 = torch.tensor([
            [0, 0, -1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -1, 0],
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(1, a, b, c) @ to3, to3 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

        to5 = torch.tensor([
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [-3**.5/3, 0, 0, 0, -3**.5/3, 0, 0, 0, 12**.5/3],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0, 0]
        ], dtype=torch.get_default_dtype())
        assert all(torch.allclose(irr_repr(2, a, b, c) @ to5, to5 @ tensor3x3_repr(a, b, c)) for a, b, c in torch.rand(10, 3))

    return to1.type(torch.get_default_dtype()), to3.type(torch.get_default_dtype()), to5.type(torch.get_default_dtype())
