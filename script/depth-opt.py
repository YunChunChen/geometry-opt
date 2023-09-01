import os
import sys

# add the repo to sys.path
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import imageio
import numpy as np
import argparse

import torch
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

    # get camera parameters
    camera_fl, _, _ = renderer.get_camera_intrinsics()
    camera_x, camera_y, camera_z = renderer.get_camera_extrinsics()

    # initialize optimizer
    optimizer = Optimizer(cfg=cfg)

    # initialize adam -> optimize camera params
    camera_params = [camera_fl, camera_x, camera_y, camera_z]
    adam_opt = torch.optim.Adam(camera_params, lr=cfg.optim.adam_lr)

    # initialize largesteps -> optimize mesh vertices
    M = compute_matrix(mesh.get_vertices(), mesh.get_faces(), lambda_=cfg.optim.lambda_)
    u = to_differential(M, mesh.get_vertices())
    u.requires_grad = True
    largesteps_opt = AdamUniform([u], lr=cfg.optim.largesteps_lr)

    updated_verts = from_differential(M, u, 'Cholesky')
    mesh.update_vertices(updated_verts)

    gt_dir = cfg.data.gt_from

    # ground truth normalized inverse depth
    # - shape: h x w
    gt_depth = torch.tensor(
        np.load('{}/gt-depth.npz'.format(gt_dir))['data'],
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
    camera_fl_list = []
    depth_list = []
    counter = 0
    for iter_idx in range(cfg.exp.num_iters):

        # render normalized inverse depth
        # - shape: h x w
        render_depth = renderer.render_depth(mesh, normalize=False)

        # render silhouette
        # - shape: h x w
        render_seg = renderer.render_silhouette(mesh)

        gt_seg_depth = gt_depth * gt_seg
        gt_seg_depth = gt_seg_depth / torch.max(gt_seg_depth)

        ren_seg_depth = render_depth * render_seg
        ren_seg_depth = ren_seg_depth / torch.max(ren_seg_depth)

        # compute the loss
        total_loss = torch.mean((gt_seg * gt_seg_depth - render_seg * ren_seg_depth)**2.0)

        # tensorboard
        writer.add_scalar('loss/depth', total_loss, iter_idx)

        print('iter:', iter_idx, 'loss:', total_loss.data.cpu().numpy())

        # back propagation
        if iter_idx < 10000:
            adam_opt.zero_grad()
            total_loss.backward()
            adam_opt.step()
        else:
            largesteps_opt.zero_grad()
            total_loss.backward()
            largesteps_opt.step()

        updated_verts = from_differential(M, u, 'Cholesky')
        mesh.update_vertices(updated_verts)

        if (iter_idx + 1) % 50 == 0:
            save_path = '{}/depth/{:06d}.png'.format(cfg.exp.save_dir, counter)
            lib.util.save_depth(ren_seg_depth, save_path, cmap=True)

            save_path = '{}/depth-diff/{:06d}.png'.format(cfg.exp.save_dir, counter)
            lib.util.save_depth(torch.abs(ren_seg_depth - gt_seg_depth), save_path, cmap=True)

            save_path = '{}/seg/{:06d}.png'.format(cfg.exp.save_dir, counter)
            lib.util.save_seg(render_seg, save_path)

            camera_fl_list.append(camera_fl.data.cpu().numpy())
            depth_list.append(render_depth.data.cpu().numpy())

            counter += 1

        if iter_idx >= 10000 and (iter_idx % 500 == 0):
            print('Remeshing...')
            mesh.remesh()
    
            M = compute_matrix(mesh.get_vertices(), mesh.get_faces(), lambda_=cfg.optim.lambda_)
            u = to_differential(M, mesh.get_vertices())
            u.requires_grad = True
            largesteps_opt = AdamUniform([u], lr=cfg.optim.largesteps_lr)

            updated_verts = from_differential(M, u, 'Cholesky')
            mesh.update_vertices(updated_verts)

        if iter_idx >= 10000 and ((iter_idx+1) % 100 == 0):
            save_path = '{}/mesh/{:06d}.obj'.format(cfg.exp.save_dir, iter_idx)
            lib.util.save_mesh(mesh, save_path)

    # make videos from images
    lib.util.image_to_video(
        image_dir='{}/depth'.format(cfg.exp.save_dir),
        save_path='{}/depth.mp4'.format(cfg.exp.save_dir)
    )

    lib.util.image_to_video(
        image_dir='{}/depth-diff'.format(cfg.exp.save_dir),
        save_path='{}/depth-diff.mp4'.format(cfg.exp.save_dir)
    )

    np.savez('{}/camera_fl.npz'.format(cfg.exp.save_dir), data=np.array(camera_fl_list))
    np.savez('{}/render_depth.npz'.format(cfg.exp.save_dir), data=np.array(depth_list))

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
    os.makedirs('{}/depth'.format(cfg.exp.save_dir), exist_ok=True)
    os.makedirs('{}/depth-diff'.format(cfg.exp.save_dir), exist_ok=True)
    os.makedirs('{}/seg'.format(cfg.exp.save_dir), exist_ok=True)
    os.makedirs('{}/mesh'.format(cfg.exp.save_dir), exist_ok=True)

    train(cfg, args)
