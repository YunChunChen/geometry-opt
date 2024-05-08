import os
import sys

# add mvcontrol to sys.path
mvcontrol_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'mvcontrol_lib'
)
sys.path.append(mvcontrol_dir)

import gc
from einops import rearrange
from rembg import remove
import numpy as np
import math

import torch
import torchvision

import xformers

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mvcontrol_lib.mvdiffusion.pipelines.pipeline_controlnet_image import MVControlNetImagePipeline
from mvcontrol_lib.mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvcontrol_lib.mvdiffusion.models.controlnet import MVControlNetModel
from mvcontrol_lib.mvdiffusion.data.normal_utils import trans_normal, img2normal

import lib.util

class MVControlNet:

    def __init__(self, cfg):
        self.cfg = cfg
        self.weight_dtype = torch.float16
        self.views = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        self.num_views = len(self.views)
        self.fix_cam_poses = self.load_fixed_poses()
        self.cond_w2c = self.fix_cam_poses['front']
        self.tgt_w2cs = [self.fix_cam_poses[view] for view in self.views]

    def load_fixed_poses(self):
        fix_cam_pose_dir = 'mvcontrol_lib/mvdiffusion/data/fixed_poses/nine_views'
        poses = {}
        for face in self.views:
            RT = np.loadtxt(os.path.join(fix_cam_pose_dir, '000_{}_RT.txt'.format(face)))
            poses[face] = RT
        return poses

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        return d_theta, d_azimuth

    @torch.no_grad()
    def get_embeddings(self):
        elevations, azimuths = [], []
        for tgt_w2c in self.tgt_w2cs:
            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, self.cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        # camera embeddings
        elevations = torch.tensor(np.array(elevations)).float().squeeze(1) # Nv
        azimuths = torch.tensor(np.array(azimuths)).float().squeeze(1) # Nv
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 6 views to train
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)
        camera_embeddings = torch.cat([camera_embeddings]*2, dim=0)

        # normal task embeddings
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)

        # color task embeddings
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)

        # task embeddings
        task_embeddings = torch.cat([normal_task_embeddings, color_task_embeddings], dim=0)

        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)
        
        return camera_task_embeddings

    @torch.no_grad()
    def generate(self, sv_rgb, sv_seg, mv_normal, mv_seg, mv_guidance_strength,  bg_color='black'):

        _, _, h, w = mv_normal.size()

        sv_rgb = (sv_rgb + 1.0) / 2.0 # 3 x 512 x 512 [0, 1]
        T = torchvision.transforms.Resize((h,w))
        sv_rgb = sv_rgb.unsqueeze(0)
        sv_seg = sv_seg.unsqueeze(0).unsqueeze(0)
        sv_rgb = lib.util.compose(sv_rgb, sv_seg, bg_color)
        sv_rgb = T(sv_rgb) # 1 x 3 x H x W

        mv_normal = mv_normal.permute(0,2,3,1) # 6 x 256 x 256 x 3
        mv_normal = mv_normal.data.cpu().numpy()
        for i in range(len(self.tgt_w2cs)):
            tgt_w2c = self.tgt_w2cs[i]
            mv_normal[i] = trans_normal(mv_normal[i], tgt_w2c, self.cond_w2c) * 0.5 + 0.5 # [0, 1]

        mv_normal = torch.tensor(mv_normal, dtype=torch.float32, device='cuda') # 6 x 256 x 256 x 3
        mv_normal = mv_normal.permute(0,3,1,2) # 6 x 3 x 256 x 256

        mv_seg = mv_seg.unsqueeze(1)

        blur_normal = lib.util.blur(mv_normal, mv_seg, bg_color)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder='image_encoder', 
            revision=self.cfg.revision
        )

        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder='feature_extractor', 
            revision=self.cfg.revision
        )

        vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder='vae', 
            revision=self.cfg.revision
        )

        unet = UNetMV2DConditionModel.from_pretrained(
            self.cfg.pretrained_unet_path, 
            subfolder='unet', 
            revision=self.cfg.revision, 
            **self.cfg.unet_from_pretrained_kwargs
        )
        
        controlnet = MVControlNetModel.from_pretrained(
            self.cfg.pretrained_controlnet_path, 
            subfolder='controlnet', 
            revision=self.cfg.revision
        )

        image_encoder.to('cuda', dtype=self.weight_dtype)
        vae.to('cuda', dtype=self.weight_dtype)
        unet.to('cuda', dtype=self.weight_dtype)
        controlnet.to('cuda', dtype=self.weight_dtype)

        pipeline = MVControlNetImagePipeline(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            scheduler=DDIMScheduler.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder='scheduler'),
            **self.cfg.pipe_kwargs
        )

        pipeline.unet.enable_xformers_memory_efficient_attention()
        pipeline.controlnet.enable_xformers_memory_efficient_attention()

        pipeline.to('cuda')

        pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=pipeline.device).manual_seed(self.cfg.seed)

        # repeat input rgb and reshape
        rgb_in = sv_rgb.repeat(2*self.num_views, 1, 1, 1) # (Nv x 2) x C x H x W

        normal_in = torch.cat([blur_normal]*2, dim=0).to(dtype=self.weight_dtype) # (Nv x 2) x C x H x W
        
        # embeddings (2B, Nv, Nce)
        camera_task_embeddings = self.get_embeddings()

        with torch.autocast('cuda'):
            guidance_scale = self.cfg.validation_guidance_scales[0]
            out = pipeline(
                rgb_in, 
                camera_task_embeddings, 
                control_image=normal_in,
                mv_guidance_strength=mv_guidance_strength,
                generator=generator, 
                guidance_scale=guidance_scale, 
                output_type='pt', 
                num_images_per_prompt=1, 
                **self.cfg.pipe_validation_kwargs
            ).images

            bsz = out.shape[0] // 2
            normals_pred = out[:bsz]
            images_pred = out[bsz:]

        del image_encoder
        del feature_extractor
        del vae
        del unet 
        del controlnet
        gc.collect()
        torch.cuda.empty_cache()

        return normals_pred, images_pred
