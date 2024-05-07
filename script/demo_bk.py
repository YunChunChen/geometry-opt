import os
import sys

# add the repo to sys.path
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import numpy as np
import argparse
import gc
from omegaconf import OmegaConf

import torch

import lib.util
import lib.config
from lib.mesh import Mesh
from lib.controlnet import ControlNet
from lib.render import Renderer

from continuous_remeshing.util.func import make_wonder3d_cameras
from continuous_remeshing.core.calc_vertex_normals import calc_vertex_normals

def main(cfg):

    # initialize mesh
    mesh = Mesh(mesh_path=cfg.mesh_path, mesh_scale=cfg.mesh_scale)

    # initialize renderer
    mv, proj = make_wonder3d_cameras()
    mv = mv[:1,:,:]
    res = cfg.sv_control_cfg.resolution
    renderer = Renderer(mv, proj, [res, res])

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # render normalized inverse depth
    # - shape: num_cameras x h x w
    render_depth = renderer.render_depth(verts, faces, normalize=True)
    print('render depth shape:', render_depth.shape)

    # save depth
    for i in range(render_depth.shape[0]):
        save_path = '{}/init_depth_{}.png'.format(cfg.save_dir, i)
        lib.util.save_depth(render_depth[i], save_path, cmap=True)

    # render normal
    # - shape: num_cameras x 3 x h x w
    render_normal = renderer.render_normal(verts, vert_normals, faces, model='ControlNet')

    print('render normal shape:', render_normal.shape)

    # save normal
    for i in range(render_normal.shape[0]):
        save_path = '{}/controlnet_normal_{}.png'.format(cfg.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

    # render normal
    # - shape: num_cameras x 3 x h x w
    render_normal = renderer.render_normal(verts, vert_normals, faces, model='Wonder3D_renderer')

    # save normal
    for i in range(render_normal.shape[0]):
        save_path = '{}/wonder3d_renderer_normal_{}.png'.format(cfg.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

    # render normal
    # - shape: num_cameras x 3 x h x w
    render_normal = renderer.render_normal(verts, vert_normals, faces, model='Wonder3D_output')

    # save normal
    for i in range(render_normal.shape[0]):
        save_path = '{}/wonder3d_output_normal_{}.png'.format(cfg.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

    # render silhouette
    # - shape: num_cameras x h x w
    render_seg = renderer.render_silhouette(verts, faces)

    # save normal
    for i in range(render_seg.shape[0]):
        save_path = '{}/seg_{}.png'.format(cfg.save_dir, i)
        lib.util.save_seg(render_seg[i], save_path)

    print('render seg shape:', render_seg.shape)

    exit()

    # RGB generation
    for num_iters in range(args.num_iters):

        # initialize controlnet
        controlnet = ControlNet(cfg)

        # generate one RGB image
        # - shape: num_samples (1) x 3 x h x w
        print("Generating RGB images...")
        rgb_image = controlnet.generate_rgb(geom_image=render_depth)

        # shape: 3 x h x w
        rgb = rgb_image.squeeze(0)

        # save rgb as png
        save_path = '{}/{:03d}-rgb.png'.format(cfg.exp.save_dir, num_iters)
        lib.util.save_rgb(rgb, save_path)

        # save rgb as npz
        save_path = '{}/{:03d}-rgb.npz'.format(cfg.exp.save_dir, num_iters)
        lib.util.save_npz(rgb, save_path)

        gc.collect()
        torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sv_cfg', default='', type=str)
    parser.add_argument('--sv_strength', default=1.0, type=float)
    parser.add_argument('--mv_cfg', default='', type=str)
    parser.add_argument('--mv_strength', default=1.0, type=float)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--save_dir', default='', type=str)
    parser.add_argument('--mesh_path', default='', type=str)
    parser.add_argument('--mesh_scale', default='', type=str)
    parser.add_argument('--prompt', default='', type=str)

    args = parser.parse_args()

    # single-view controlnet config
    schema = OmegaConf.structured(lib.config.SVControlConfig)
    sv_cfg = OmegaConf.load(args.sv_cfg)
    sv_cfg = OmegaConf.merge(schema, sv_cfg)
    sv_cfg.strength = args.sv_strength
    sv_cfg.prompt = args.prompt

    # all config
    cfg = OmegaConf.structured(lib.config.Config)
    cfg.sv_control_cfg = sv_cfg
    cfg.save_dir = args.save_dir
    cfg.mesh_path = args.mesh_path
    cfg.mesh_scale = args.mesh_scale

    # save dir
    os.makedirs(cfg.save_dir, exist_ok=True)

    # save config
    cfg_path = os.path.join(cfg.save_dir, 'config.yaml')
    OmegaConf.save(cfg, cfg_path)

    main(cfg)
