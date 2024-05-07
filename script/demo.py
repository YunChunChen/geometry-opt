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
import torch.nn.functional as F

import lib.util
import lib.config
from lib.mesh import Mesh
from lib.renderer import Renderer
from lib.svcontrolnet import SVControlNet
from lib.mvcontrolnet import MVControlNet

from meshopt_lib.util.func import make_wonder3d_cameras
from meshopt_lib.core.opt import MeshOptimizer

def main(cfg):

    # initialize mesh
    mesh = Mesh(mesh_path=cfg.mesh_path, mesh_scale=cfg.mesh_scale)

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # --- single-view RGB generation

    # initialize renderer
    mv, proj = make_wonder3d_cameras()
    mv = mv[:1,:,:]
    sv_res = cfg.sv_control_cfg.resolution
    sv_renderer = Renderer(mv, proj, [sv_res, sv_res])

    # render normalized inverse depth
    # - shape: 1 x h x w
    render_sv_depth = sv_renderer.render_depth(verts, faces, normalize=True)

    # save depth
    save_path = '{}/init_depth.png'.format(cfg.save_dir)
    lib.util.save_depth(render_sv_depth[0], save_path, cmap=True)

    # initialize single-view controlnet
    svcontrolnet = SVControlNet(cfg.sv_control_cfg)

    # generate one RGB image
    # - shape: num_samples (1) x 3 x h x w [-1, 1]
    print("Generating RGB images from depth renderings...")
    sv_rgb = svcontrolnet.generate_rgb(geom_image=render_sv_depth)

    # save rgb as png
    save_path = '{}/sv_rgb.png'.format(cfg.save_dir)
    lib.util.save_rgb(sv_rgb[0], save_path)

    # initialize segment anything
    seg_predictor = lib.util.sam_init()

    # predict segmentation
    sv_rgb_cpu = sv_rgb[0].permute(1,2,0).data.cpu().numpy() * 127.5 + 127.5
    sv_bbox = lib.util.pred_bbox(sv_rgb_cpu)
    sv_mask = lib.util.pred_seg(seg_predictor, sv_rgb_cpu, sv_bbox) # h x w

    # save segmentation as png
    save_path = '{}/sv_seg.png'.format(cfg.save_dir)
    lib.util.save_seg(sv_mask, save_path)

    # --- multi-view RGB & normal generation

    # initialize renderer
    mv, proj = make_wonder3d_cameras()
    mv_res = cfg.mv_control_cfg.resolution
    mv_renderer = Renderer(mv, proj, [mv_res, mv_res])

    # render normal in camera frame
    # - shape: num_views x 3 x h x w [-1, 1]
    render_mv_normal = mv_renderer.render_normal(verts, faces, model='Wonder3D_renderer')

    # render silhouette
    # - shape: num_views x h x w [0, 1]
    render_mv_seg = mv_renderer.render_silhouette(verts, faces)

    # save multi-view normal & seg renderings
    for i in range(render_mv_normal.shape[0]):
        save_path = '{}/init_normal_{}.png'.format(cfg.save_dir, i)
        lib.util.save_rgb(render_mv_normal[i], save_path)

        save_path = '{}/init_seg_{}.png'.format(cfg.save_dir, i)
        lib.util.save_seg(render_mv_seg[i], save_path)

    # initialize multi-view controlnet
    mvcontrolnet = MVControlNet(cfg.mv_control_cfg)

    # generate multi-view RGBs and normals
    # - shape: num_views x 3 x H x W
    print("Generating multi-view RGBs and normals...")
    mv_normals, mv_rgbs = mvcontrolnet.generate(sv_rgb=sv_rgb[0], sv_seg=sv_mask, mv_normal=render_mv_normal, mv_seg=render_mv_seg)

    # predict segmentation
    mv_rgbs_cpu = mv_rgbs.permute(0,2,3,1).data.cpu().numpy() * 255
    mv_masks = []
    for i in range(mv_rgbs.shape[0]):
        mv_bbox = lib.util.pred_bbox(mv_rgbs_cpu[i])
        mv_mask = lib.util.pred_seg(seg_predictor, mv_rgbs_cpu[i], mv_bbox) # h x w
        mv_masks.append(mv_mask.data.cpu().numpy())

    # num_views x h x w
    mv_masks = torch.tensor(np.array(mv_masks), dtype=torch.float32, device='cuda')

    # save multi-view normal & rgb & segmentation predictions
    for i in range(mv_normals.shape[0]):
        save_path = '{}/mv_normal_{}.png'.format(cfg.save_dir, i)
        lib.util.save_image_wonder3d(mv_normals[i], save_path)

        save_path = '{}/mv_rgb_{}.png'.format(cfg.save_dir, i)
        lib.util.save_image_wonder3d(mv_rgbs[i], save_path)

        save_path = '{}/mv_seg_{}.png'.format(cfg.save_dir, i)
        lib.util.save_seg(mv_masks[i], save_path)

    # --- mesh optimization
    gt_normals = mv_normals * 2.0 - 1.0
    gt_normals = F.interpolate(gt_normals, scale_factor=2.0, mode='bilinear', antialias=False).clamp(-1.0, 1.0)
    gt_normals = F.normalize(gt_normals, dim=1)

    gt_masks = F.interpolate(mv_masks.unsqueeze(1), scale_factor=2.0, mode='bilinear', antialias=False).clamp(0.0, 1.0)

    #gt_masks = gt_masks.unsqueeze(1) # num_views x 1 x h x w

    gt_normals = gt_normals * gt_masks
    gt_masks = gt_masks.squeeze(1)

    for i in range(gt_normals.shape[0]):
        save_path = '{}/gt_normal_{}.png'.format(args.save_dir, i)
        lib.util.save_rgb(gt_normals[i], save_path)

        save_path = '{}/gt_seg_{}.png'.format(args.save_dir, i)
        lib.util.save_seg(gt_masks[i], save_path)

    gt_image = torch.cat([gt_normals, gt_masks.unsqueeze(1)], dim=1) 

    gt_image_dx, gt_image_dy = lib.util.dxdy_normal_image(gt_normals)

    opt = MeshOptimizer(verts, faces)
    verts = opt.vertices

    # initialize renderer
    mv, proj = make_wonder3d_cameras()
    mv_res = cfg.sv_control_cfg.resolution
    mv_renderer = Renderer(mv, proj, [mv_res, mv_res])

    for i in range(cfg.num_iters):
        opt.zero_grad()

        # render normal
        # - shape: num_views x 3 x h x w
        render_normal = mv_renderer.render_normal(verts, faces, model='Wonder3D_output', shading='flat')

        # render silhouette
        # - shape: num_views x 1 x h x w
        render_seg = mv_renderer.render_silhouette(verts, faces, shading='flat')

        render_image = torch.cat([render_normal, render_seg.unsqueeze(1)], dim=1)

        loss = (render_image - gt_image).abs().mean()
    
        if i > cfg.num_iters//2:
            render_image_dx, render_image_dy = lib.util.dxdy_normal_image(render_normal)
            loss = loss + (render_image_dx - gt_image_dx).abs().mean() + (render_image_dy - gt_image_dy).abs().mean()

        print('loss:', loss.data.cpu().numpy())

        loss.backward()

        opt.step()

        verts, faces = opt.remesh()

    # save final normal / seg
    for i in range(render_normal.shape[0]):
        save_path = '{}/final_normal_{}.png'.format(cfg.save_dir, i)
        lib.util.save_rgb(render_normal[i], save_path)

        save_path = '{}/final_seg_{}.png'.format(cfg.save_dir, i)
        lib.util.save_seg(render_seg[i], save_path)

    import igl
    # save final mesh
    save_path = '{}/final_mesh.obj'.format(cfg.save_dir)
    igl.write_triangle_mesh(
        save_path, 
        verts.data.cpu().numpy() / cfg.mesh_scale,
        faces.data.cpu().numpy()
    )

    import trimesh
    # save as glb
    glb_path = '{}/final_mesh.glb'.format(cfg.save_dir)
    final_mesh = trimesh.load(save_path)
    s = trimesh.Scene([final_mesh])
    s.export(file_obj=glb_path)

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

    # multi-view controlnet config
    schema = OmegaConf.structured(lib.config.MVControlConfig)
    mv_cfg = OmegaConf.load(args.mv_cfg)
    mv_cfg = OmegaConf.merge(schema, mv_cfg)
    mv_cfg.strength = args.mv_strength

    # all config
    cfg = OmegaConf.structured(lib.config.Config)
    cfg.sv_control_cfg = sv_cfg
    cfg.mv_control_cfg = mv_cfg
    cfg.save_dir = args.save_dir
    cfg.mesh_path = args.mesh_path
    cfg.mesh_scale = args.mesh_scale
    cfg.num_iters = args.num_iters

    # save dir
    os.makedirs(cfg.save_dir, exist_ok=True)

    # save config
    cfg_path = os.path.join(cfg.save_dir, 'config.yaml')
    OmegaConf.save(cfg, cfg_path)

    main(cfg)
