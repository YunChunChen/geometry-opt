import os
import sys

# add the repo to sys.path
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import imageio
import numpy as np
import argparse
import gc

import torch

import lib.util
from lib.mesh import Mesh
from lib.controlnet import ControlNet
from lib.config import get_cfg_defaults

from continuous_remeshing.util.render import Renderer
from continuous_remeshing.util.func import make_star_cameras, make_wonder3d_cameras
from continuous_remeshing.core.calc_vertex_normals import calc_vertex_normals

def main(cfg, args):

    # initialize mesh
    mesh = Mesh(cfg)

    # perspective projection
    #mv, proj = make_star_cameras(1,1)
    mv, proj = make_wonder3d_cameras(1,1)

    # initialize renderer
    res = cfg.renderer.resolution
    renderer = Renderer(mv, proj, [res,res])

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # calculate vertex normals
    vert_normals = calc_vertex_normals(verts, faces)

    # render normalized inverse depth
    # - shape: h x w
    render_depth = renderer.render_depth(verts, faces, normalize=True).squeeze(0)

    # save depth
    for i in range(6):
        save_path = '{}/init_depth_{}.png'.format(cfg.exp.save_dir, i)
        lib.util.save_depth(render_depth[i], save_path, cmap=True)

    render_depth = render_depth[0,:,:]

    # render normal
    # - shape: 3 x h x w
    render_normal = renderer.render_normal(verts, vert_normals, faces, model='ControlNet').squeeze(0)

    # save normal
    for i in range(6):
        save_path = '{}/controlnet_normal_{}.png'.format(cfg.exp.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

    render_normal = render_normal[0,:,:]

    # render normal
    # - shape: 3 x h x w
    render_normal = renderer.render_normal(verts, vert_normals, faces, model='Wonder3D_renderer').squeeze(0)

    # save normal
    for i in range(6):
        save_path = '{}/wonder3d_renderer_normal_{}.png'.format(cfg.exp.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

    # render normal
    # - shape: 3 x h x w
    render_normal = renderer.render_normal(verts, vert_normals, faces, model='Wonder3D_output').squeeze(0)

    # save normal
    for i in range(6):
        save_path = '{}/wonder3d_output_normal_{}.png'.format(cfg.exp.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

    exit()

    # RGB generation
    for num_iters in range(args.num_iters):

        # initialize controlnet
        controlnet = ControlNet(cfg)

        # generate one RGB image
        # - shape: num_samples (1) x 3 x h x w
        print("Generating RGB images...")
        if cfg.controlnet.input_mode == 'depth':
            rgb_image = controlnet.generate_rgb(geom_image=render_depth)
        elif cfg.controlnet.input_mode == 'normal':
            rgb_image = controlnet.generate_rgb(geom_image=render_normal)

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

        # predict segmentation
        # - shape: num_masks (3) x h x w
        #print("Predicting segmentation...")
        #gt_seg = controlnet.predict_segmentation(rgb)

        # save ground truth segmentation
        #save_path = '{}/{:03d}-seg-0.png'.format(cfg.exp.save_dir, data_idx)
        #lib.util.save_seg(gt_seg[0], save_path)

        #save_path = '{}/{:03d}-seg-1.png'.format(cfg.exp.save_dir, data_idx)
        #lib.util.save_seg(gt_seg[1], save_path)

        #save_path = '{}/{:03d}-seg-2.png'.format(cfg.exp.save_dir, data_idx)
        #lib.util.save_seg(gt_seg[2], save_path)

        '''
        # normal prediction
        # - shape: 3 x h x w
        print("Predicting OmniData normal...")
        pred_normal = controlnet.predict_normal_omnidata(rgb_image).squeeze(0)

        # convert coordinate convention
        pred_normal = pred_normal * -1.0

        # save predicted normal
        save_path = '{}/{:03d}-omnidata-normal.png'.format(cfg.exp.save_dir, num_iters)
        lib.util.save_rgb(pred_normal, save_path)

        # save predicted normal as npz
        save_path = '{}/{:03d}-omnidata-normal.npz'.format(cfg.exp.save_dir, num_iters)
        lib.util.save_npz(pred_normal, save_path)

        gc.collect()
        torch.cuda.empty_cache()

        # normal prediction
        # - shape: 3 x h x w
        print("Predicting DSINE normal...")
        pred_normal = controlnet.predict_normal_dsine(rgb_image).squeeze(0)

        # save predicted normal
        save_path = '{}/{:03d}-dsine-normal.png'.format(cfg.exp.save_dir, num_iters)
        lib.util.save_rgb(pred_normal, save_path)

        # save predicted normal as npz
        save_path = '{}/{:03d}-dsine-normal.npz'.format(cfg.exp.save_dir, num_iters)
        lib.util.save_npz(pred_normal, save_path)

        gc.collect()
        torch.cuda.empty_cache()
        '''

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', type=str)
    parser.add_argument('--strength', default=1.0, type=float)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--save_dir', default='', type=str)
    parser.add_argument('--input_mode', default='', type=str)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    cfg.exp.num_iters = args.num_iters
    cfg.controlnet.strength = args.strength
    cfg.controlnet.input_mode = args.input_mode

    cfg.exp.save_dir = args.save_dir
    
    cfg.freeze()
    
    # output
    os.makedirs(cfg.exp.save_dir, exist_ok=True)

    main(cfg, args)
