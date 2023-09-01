import os
import sys

# add the repo to sys.path
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import imageio
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import lib.util
from lib.mesh import Mesh
from lib.controlnet import ControlNet
from lib.renderer import Renderer
from lib.optimizer import Optimizer
from lib.config import get_cfg_defaults

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

def train(cfg, args):

    # initialize mesh
    mesh = Mesh(cfg)

    # initialize renderer
    renderer = Renderer(cfg)

    # initialize largesteps -> optimize mesh vertices
    M = compute_matrix(mesh.get_vertices(), mesh.get_faces(), lambda_=cfg.optim.lambda_)
    u = to_differential(M, mesh.get_vertices())
    u.requires_grad = True
    largesteps_opt = AdamUniform([u], lr=cfg.optim.largesteps_lr)

    updated_verts = from_differential(M, u, 'Cholesky')
    mesh.update_vertices(updated_verts)

    gt_dir = cfg.data.gt_from

    # ground truth normal
    # - shape: 3 x h x w
    gt_normal = torch.tensor(
        np.load('{}/gt-normal.npz'.format(gt_dir))['data'],
        dtype=torch.float32,
        device='cuda'
    )

    # ground truth segmentation
    # - shape: h x w
    gt_seg = torch.tensor(
        imageio.imread('{}/gt-seg.png'.format(gt_dir)),
        dtype=torch.float32, 
        device='cuda'
    ) / 255.0

    # -- start optimizing 

    # tensorboard
    writer = SummaryWriter(log_dir='{}/tb'.format(cfg.exp.save_dir))

    # optimization
    counter = 0
    for iter_idx in range(cfg.exp.num_iters):

        # render normal
        # - shape: 3 x h x w
        render_normal = renderer.render_normal(mesh, antialias=True)

        # render silhouette
        # - shape: h x w
        render_seg = renderer.render_silhouette(mesh)

        gt_seg_normal = gt_normal * gt_seg

        ren_seg_normal = render_normal * render_seg

        # compute the loss
        total_loss = torch.mean(1.0 - F.cosine_similarity(ren_seg_normal, gt_seg_normal, dim=0))

        # tensorboard
        writer.add_scalar('loss/normal', total_loss, iter_idx)

        print('iter:', iter_idx, 'loss:', total_loss.data.cpu().numpy())

        # back propagation
        largesteps_opt.zero_grad()
        total_loss.backward()
        largesteps_opt.step()

        updated_verts = from_differential(M, u, 'Cholesky')
        mesh.update_vertices(updated_verts)

        if (iter_idx + 1) % 50 == 0:
            save_path = '{}/normal/{:06d}.png'.format(cfg.exp.save_dir, counter)
            lib.util.save_rgb(ren_seg_normal, save_path)

            save_path = '{}/mesh/{:06d}.obj'.format(cfg.exp.save_dir, counter)
            lib.util.save_mesh(mesh, save_path)

            counter += 1

        if (iter_idx + 1) % 500 == 0:
            print('Remeshing...')
            mesh.remesh()
    
            M = compute_matrix(mesh.get_vertices(), mesh.get_faces(), lambda_=cfg.optim.lambda_)
            u = to_differential(M, mesh.get_vertices())
            u.requires_grad = True
            largesteps_opt = AdamUniform([u], lr=cfg.optim.largesteps_lr)

            updated_verts = from_differential(M, u, 'Cholesky')
            mesh.update_vertices(updated_verts)

    # make videos from images
    lib.util.image_to_video(
        image_dir='{}/normal'.format(cfg.exp.save_dir),
        save_path='{}/normal.mp4'.format(cfg.exp.save_dir)
    )

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', type=str)
    parser.add_argument('--strength', default=1.0, type=float)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--save_dir', default='', type=str)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    cfg.exp.num_iters = args.num_iters
    cfg.controlnet.strength = args.strength

    cfg.exp.save_dir = args.save_dir
    
    cfg.freeze()
    
    # output
    os.makedirs(cfg.exp.save_dir, exist_ok=True)
    os.makedirs('{}/normal'.format(cfg.exp.save_dir), exist_ok=True)
    os.makedirs('{}/mesh'.format(cfg.exp.save_dir), exist_ok=True)

    train(cfg, args)
