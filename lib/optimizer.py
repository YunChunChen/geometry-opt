import torch

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

class Optimizer:

    def __init__(self, cfg):
        self.cfg = cfg

    def init_largesteps(self, mesh):
        verts = mesh.get_vertices()
        faces = mesh.get_faces()
        M = compute_matrix(verts, faces, lambda_=self.cfg.optim.lambda_)
        u = to_differential(M, verts)
        u.requires_grad = True
        optimizer = AdamUniform([u], lr=self.cfg.optim.largesteps_lr)
        return M, u, optimizer

    def update_largesteps(self, M, u, mesh):
        # update mesh vertices
        updated_verts = from_differential(M, u, 'Cholesky')
        mesh.update_vertices(updated_verts)
        return 

    def init_adam(self, params):
        optimizer = torch.optim.Adam(params, lr=self.cfg.optim.adam_lr)
        return optimizer
