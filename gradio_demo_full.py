import gradio as gr
import os

import numpy as np
import argparse
import gc
from omegaconf import OmegaConf
import igl
import trimesh
import aspose.threed as a3d

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

save_dir = 'debug'
os.makedirs(save_dir, exist_ok=True)

mesh_scale = 1.0

# initialize single-view renderer
mv, proj = make_wonder3d_cameras()
mv = mv[:1,:,:]
sv_res = 512
sv_renderer = Renderer(mv, proj, [sv_res, sv_res])

# initialize segment anything
seg_predictor = lib.util.sam_init()

# initialize multi-view renderer
mv, proj = make_wonder3d_cameras()
mv_res = 256
mv_renderer = Renderer(mv, proj, [mv_res, mv_res])

# multi-view controlnet config
schema = OmegaConf.structured(lib.config.MVControlConfig)
mv_cfg = OmegaConf.load('config/mv_control_cfg.yaml')
mv_cfg = OmegaConf.merge(schema, mv_cfg)

# initialize multi-view controlnet
mvcontrolnet = MVControlNet(mv_cfg)

# initialize renderer for mesh optimization
mv, proj = make_wonder3d_cameras()
mv_res = 512
opt_renderer = Renderer(mv, proj, [mv_res, mv_res])
num_iters = 2000

def run_convert_to_glb(mesh_path):
    # initialize mesh
    mesh = Mesh(mesh_path=mesh_path, mesh_scale=mesh_scale)

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # get normalization scale factor
    global normalize_factor
    normalize_factor = float(mesh.get_normalize_scale().data.cpu().numpy())

    # save as obj
    obj_path = '{}/input_mesh.obj'.format(save_dir)
    igl.write_triangle_mesh(
        obj_path,
        verts.data.cpu().numpy() / mesh_scale,
        faces.data.cpu().numpy()
    )

    # save as glb
    scene = a3d.Scene.from_file(obj_path)
    glb_path = '{}/input_mesh.glb'.format(save_dir)
    scene.save(glb_path)

    return glb_path

def run_segmentation(rgb):
    bbox = lib.util.pred_bbox(rgb)
    mask = lib.util.pred_seg(seg_predictor, rgb, bbox) # h x w

    out_image = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = rgb
    out_image[:,:,3] = (mask.data.cpu().numpy() * 255).astype(np.uint8)

    mask_np = mask.data.cpu().numpy()
    np.savez('{}/mask.npz'.format(save_dir), data=mask_np)

    return out_image

def run_render_depth(mesh_path):
    mesh_path = '{}/input_mesh.obj'.format(save_dir)
    # initialize mesh
    mesh = Mesh(mesh_path=mesh_path, mesh_scale=mesh_scale)

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # render normalized inverse depth
    # - shape: 1 x h x w
    render_sv_depth = sv_renderer.render_depth(verts, faces, normalize=True)

    render_sv_depth_cpu = render_sv_depth[0].mul(255).clamp(0, 255).to('cpu', torch.uint8).numpy()

    render_sv_depth_np = render_sv_depth.data.cpu().numpy()
    np.savez('{}/depth.npz'.format(save_dir), data=render_sv_depth_np)

    return render_sv_depth_cpu

def run_single_view_generation(prompt, sv_strength_slider):
    render_depth_np = np.load('{}/depth.npz'.format(save_dir))['data']
    render_sv_depth = torch.tensor(render_depth_np, dtype=torch.float32, device='cuda')

    # single-view controlnet
    schema = OmegaConf.structured(lib.config.SVControlConfig)
    sv_cfg = OmegaConf.load('config/sv_control_cfg.yaml')
    sv_cfg = OmegaConf.merge(schema, sv_cfg)
    sv_cfg.strength = 1.0
    sv_cfg.prompt = prompt

    svcontrolnet = SVControlNet(sv_cfg)

    guidance_strength = sv_strength_slider / 100.0

    # generate RGB images
    # - shape: num_samples x 3 x h x w [-1, 1]
    print("Generating RGB images from depth renderings...")
    sv_rgb = svcontrolnet.generate_rgb(geom_image=render_sv_depth, guidance_strength=guidance_strength)

    sv_rgb_cpu = sv_rgb[0].mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to('cpu', torch.uint8).numpy()

    sv_rgb_np = sv_rgb.data.cpu().numpy()
    np.savez('{}/sv_rgb.npz'.format(save_dir), data=sv_rgb_np)

    return sv_rgb_cpu

def run_multi_view_generation(mesh_path, mv_strength_slider):
    mesh_path = '{}/input_mesh.obj'.format(save_dir)
    # initialize mesh
    mesh = Mesh(mesh_path=mesh_path, mesh_scale=mesh_scale)

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # render normal in camera frame
    # - shape: num_views x 3 x h x w [-1, 1]
    render_mv_normal = mv_renderer.render_normal(verts, faces, model='Wonder3D_renderer')

    # render silhouette
    # - shape: num_views x h x w [0, 1]
    render_mv_seg = mv_renderer.render_silhouette(verts, faces)

    # read in single-view rgb
    sv_rgb_np = np.load('{}/sv_rgb.npz'.format(save_dir))['data']
    sv_rgb = torch.tensor(sv_rgb_np, dtype=torch.float32, device='cuda')

    # read in single-view mask
    mask_np = np.load('{}/mask.npz'.format(save_dir))['data']
    sv_mask = torch.tensor(mask_np, dtype=torch.float32, device='cuda')

    guidance_strength = mv_strength_slider / 100.0

    # generate multi-view RGBs and normals
    # - shape: num_views x 3 x H x W
    print("Generating multi-view RGBs and normals...")
    mv_normals, mv_rgbs = mvcontrolnet.generate(
        sv_rgb=sv_rgb[0], 
        sv_seg=sv_mask, 
        mv_normal=render_mv_normal, 
        mv_seg=render_mv_seg,
        mv_guidance_strength=guidance_strength
    )

    # predict segmentation
    mv_rgbs_cpu = mv_rgbs.permute(0,2,3,1).data.cpu().numpy() * 255
    mv_masks = []
    for i in range(mv_rgbs.shape[0]):
        mv_bbox = lib.util.pred_bbox(mv_rgbs_cpu[i])
        mv_mask = lib.util.pred_seg(seg_predictor, mv_rgbs_cpu[i], mv_bbox) # h x w
        mv_masks.append(mv_mask.data.cpu().numpy())

    mv_rgbs_cpu = mv_rgbs.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).numpy()
    mv_normals_cpu = mv_normals.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).numpy()

    # num_views x h x w
    mv_masks = np.array(mv_masks)
    np.savez('{}/mv_masks.npz'.format(save_dir), data=mv_masks)

    mv_normals_np = mv_normals.data.cpu().numpy()
    np.savez('{}/mv_normals.npz'.format(save_dir), data=mv_normals_np)

    return_list = []
    for i in range(mv_rgbs_cpu.shape[0]):
        return_list.append(mv_rgbs_cpu[i])
    for i in range(mv_normals_cpu.shape[0]):
        return_list.append(mv_normals_cpu[i])

    return return_list

def run_mesh_opt(mesh_path):
    mesh_path = '{}/input_mesh.obj'.format(save_dir)
    # initialize mesh
    mesh = Mesh(mesh_path=mesh_path, mesh_scale=mesh_scale)

    # get mesh vertices and faces
    verts = mesh.get_vertices()
    faces = mesh.get_faces()

    # load gt normal and seg
    mv_normal_np = np.load('{}/mv_normals.npz'.format(save_dir))['data']
    mv_normal = torch.tensor(mv_normal_np, dtype=torch.float32, device='cuda')

    mv_seg_np = np.load('{}/mv_masks.npz'.format(save_dir))['data']
    mv_seg = torch.tensor(mv_seg_np, dtype=torch.float32, device='cuda')

    gt_normal = mv_normal * 2.0 - 1.0
    gt_normal = F.interpolate(gt_normal, scale_factor=2.0, mode='bilinear', antialias=False).clamp(-1.0, 1.0)
    gt_normal = F.normalize(gt_normal, dim=1)
    gt_seg = F.interpolate(mv_seg.unsqueeze(1), scale_factor=2.0, mode='bilinear', antialias=False).clamp(0.0, 1.0)

    gt_normal = gt_normal * gt_seg

    gt_image = torch.cat([gt_normal, gt_seg], dim=1)

    gt_image_dx, gt_image_dy = lib.util.dxdy_normal_image(gt_normal)

    opt = MeshOptimizer(verts, faces)
    verts = opt.vertices

    for i in range(num_iters):
        opt.zero_grad()

        # render normal
        # - shape: num_views x 3 x h x w
        render_normal = opt_renderer.render_normal(verts, faces, model='Wonder3D_output', shading='flat')

        # render silhouette
        # - shape: num_views x 1 x h x w
        render_seg = opt_renderer.render_silhouette(verts, faces, shading='flat')

        render_image = torch.cat([render_normal, render_seg.unsqueeze(1)], dim=1)

        loss = (render_image - gt_image).abs().mean()

        if i > num_iters//2:
            render_image_dx, render_image_dy = lib.util.dxdy_normal_image(render_normal)
            loss = loss + (render_image_dx - gt_image_dx).abs().mean() + (render_image_dy - gt_image_dy).abs().mean()

        loss.backward()

        opt.step()

        verts, faces = opt.remesh()

    # save final mesh
    obj_path = '{}/final_mesh.obj'.format(save_dir)
    igl.write_triangle_mesh(
        obj_path,
        verts.data.cpu().numpy() / mesh_scale * normalize_factor,
        faces.data.cpu().numpy()
    )

    glb_path = '{}/final_mesh.glb'.format(save_dir)
    final_mesh = trimesh.load(obj_path)
    s = trimesh.Scene([final_mesh])
    s.export(file_obj=glb_path)

    return glb_path

def run_download_mesh():
    obj_path = '{}/final_mesh.obj'.format(save_dir)
    return obj_path 

with gr.Blocks() as demo:
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            input_mesh = gr.Model3D(label="Input mesh", height=240, camera_position=(90, None, None))
            input_text = gr.Textbox(label='Prompt')
            run_upload_btn = gr.Button("Upload", variant='primary', interactive=True)

        with gr.Column(scale=1):
            render_depth = gr.Image(interactive=False, image_mode='L', label="Depth rendering")
            sv_strength_slider = gr.Slider(0, 100, value=75, step=1, label="Single-view guidance strength (%)")
            run_sv_btn = gr.Button("Single-view generation", variant='primary', interactive=True)

        with gr.Column(scale=1):
            sv_rgb = gr.Image(interactive=False, label="Single-view RGB")
            run_seg_btn = gr.Button("Remove background", variant='primary', interactive=True)

        with gr.Column(scale=1):
            sv_rgb_seg = gr.Image(interactive=False, label="Single-view RGB w/o background")
            mv_strength_slider = gr.Slider(0, 100, value=75, step=1, label="Multi-view guidance strength (%)")
            run_mv_btn = gr.Button("Multi-view generation", variant='primary', interactive=True)

    with gr.Row():
        rgb_0 = gr.Image(interactive=False, height=240, show_label=False)
        rgb_1 = gr.Image(interactive=False, height=240, show_label=False)
        rgb_2 = gr.Image(interactive=False, height=240, show_label=False)
        rgb_3 = gr.Image(interactive=False, height=240, show_label=False)
        rgb_4 = gr.Image(interactive=False, height=240, show_label=False)
        rgb_5 = gr.Image(interactive=False, height=240, show_label=False)

    with gr.Row():
        normal_0 = gr.Image(interactive=False, height=240, show_label=False)
        normal_1 = gr.Image(interactive=False, height=240, show_label=False)
        normal_2 = gr.Image(interactive=False, height=240, show_label=False)
        normal_3 = gr.Image(interactive=False, height=240, show_label=False)
        normal_4 = gr.Image(interactive=False, height=240, show_label=False)
        normal_5 = gr.Image(interactive=False, height=240, show_label=False)

    with gr.Row():
        final_mesh = gr.Model3D(label="Output mesh", height=240, camera_position=(90, None, None))
        run_opt_btn = gr.Button("Mesh optimization", variant='primary', interactive=True)

    run_upload_btn.click(
        fn=run_convert_to_glb,
        inputs=[input_mesh],
        outputs=[input_mesh],
        queue=True                        
    ).success(
        fn=run_render_depth,
        inputs=[input_mesh],
        outputs=[render_depth],
        queue=True                        
    )

    run_sv_btn.click(
        fn=run_single_view_generation,
        inputs=[input_text, sv_strength_slider],
        outputs=[sv_rgb],
        queue=True
    )
    
    run_seg_btn.click(
        fn=run_segmentation,
        inputs=[sv_rgb],
        outputs=[sv_rgb_seg],
        queue=True
    )

    run_mv_btn.click(
        fn=run_multi_view_generation,
        inputs=[input_mesh, mv_strength_slider],
        outputs=[rgb_0, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, normal_0, normal_1, normal_2, normal_3, normal_4, normal_5],
        queue=True
    )

    run_opt_btn.click(
        fn=run_mesh_opt,
        inputs=[input_mesh],
        outputs=[final_mesh],
        queue=True
    )

    demo.queue().launch(share=True, max_threads=10)
