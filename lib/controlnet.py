import os
import sys

# add controlnet_repo to sys.path
controlnet_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'controlnet_lib'
)
sys.path.append(controlnet_dir)

import gc
import numpy as np

import torch

import controlnet_lib.config
from controlnet_lib.share import *
from controlnet_lib.cldm.model import create_model, load_state_dict
from controlnet_lib.cldm.ddim_hacked import DDIMSampler
from controlnet_lib.annotator.midas import MidasDetector
from controlnet_lib.annotator.normalbae import NormalBaeDetector

from segment_anything import sam_model_registry, SamPredictor

class ControlNet:
    
    def __init__(self, cfg):
        self.num_samples = cfg.controlnet.num_samples
        self.ddim_steps = cfg.controlnet.ddim_steps
        self.eta = cfg.controlnet.eta
        self.scale = cfg.controlnet.scale
        self.guess_mode = cfg.controlnet.guess_mode
        self.strength = cfg.controlnet.strength
        self.from_depth = cfg.controlnet.from_depth
        self.from_normal = cfg.controlnet.from_normal
        self.prompt = cfg.controlnet.prompt
        self.a_prompt = cfg.controlnet.a_prompt
        self.n_prompt = cfg.controlnet.n_prompt

    def get_depth_network(self):
        model = create_model(
            'controlnet_lib/models/control_v11f1p_sd15_depth.yaml'
        ).cpu()
        model.load_state_dict(
            load_state_dict('controlnet_lib/models/v1-5-pruned.ckpt', location='cuda'), 
            strict=False
        )
        model.load_state_dict(
            load_state_dict('controlnet_lib/models/control_v11f1p_sd15_depth.pth', location='cuda'), 
            strict=False
        )
        model = model.cuda()
        return model

    def get_normal_network(self):
        model = create_model(
            'controlnet_lib/models/control_v11p_sd15_normalbae.yaml'
        ).cpu()
        model.load_state_dict(
            load_state_dict('controlnet_lib/models/v1-5-pruned.ckpt', location='cuda'), 
            strict=False
        )
        model.load_state_dict(
            load_state_dict('controlnet_lib/models/control_v11p_sd15_normalbae.pth', location='cuda'), 
            strict=False
        )
        model = model.cuda()
        return model

    def get_ddim_sampler(self, model):
        ddim_sampler = DDIMSampler(model)
        return ddim_sampler

    @torch.no_grad()
    def generate_rgb(self, geom_image):

        if self.from_depth:
            controlnet = self.get_depth_network()
            geom_image = geom_image.unsqueeze(0).unsqueeze(0) # h x w -> 1 x 1 x h x w
            geom_image = geom_image.repeat(self.num_samples, 3, 1, 1)
        elif self.from_normal:
            controlnet = self.get_normal_network()
            geom_image = geom_image.unsqueeze(0) # 3 x h x w -> 1 x 3 x h x w
            geom_image = geom_image.repeat(self.num_samples, 1, 1, 1)

        ddim_sampler = self.get_ddim_sampler(controlnet)

        if controlnet_lib.config.save_memory:
            controlnet.low_vram_shift(is_diffusing=False)
        
        cond = {
            'c_concat': [geom_image], 
            'c_crossattn': [controlnet.get_learned_conditioning(
                [self.prompt + ', ' + self.a_prompt] * self.num_samples
            )]
        }

        un_cond = {
            'c_concat': None if self.guess_mode else [geom_image], 
            'c_crossattn': [controlnet.get_learned_conditioning([self.n_prompt] * self.num_samples)]
        }

        if controlnet_lib.config.save_memory:
            controlnet.low_vram_shift(is_diffusing=True)

        controlnet.control_scales = [
            self.strength * (0.825 ** float(12 - i)) for i in range(13)
        ] if self.guess_mode else ([self.strength] * 13)

        _, _, H, W = geom_image.size()

        shape = (4, H // 8, W // 8)

        samples, intermediates = ddim_sampler.sample(
            self.ddim_steps, 
            self.num_samples,
            shape, 
            cond, 
            verbose=False, 
            eta=self.eta,
            unconditional_guidance_scale=self.scale,
            unconditional_conditioning=un_cond
        )

        if controlnet_lib.config.save_memory:
            controlnet.low_vram_shift(is_diffusing=False)

        rgb_images = controlnet.decode_first_stage(samples) # batch x 3 x h x w

        # delete controlnet and release GPU memory
        del controlnet
        gc.collect()
        torch.cuda.empty_cache()
        print("Releasing controlnet GPU memory...")

        return rgb_images

    @torch.no_grad()
    def predict_depth(self, rgb_images):
        depth_net = MidasDetector()
        depth_images = depth_net(rgb_images)

        # delete depth network and release GPU memory
        del depth_net
        gc.collect()
        torch.cuda.empty_cache()
        print("Releasing depth net GPU memory...")

        return depth_images

    @torch.no_grad()
    def predict_normal(self, rgb_images):
        normal_net = NormalBaeDetector()
        rgb_images = (rgb_images + 1.0) / 2.0
        normal_images = normal_net(rgb_images)

        # delete normal network and release GPU memory
        del normal_net
        gc.collect()
        torch.cuda.empty_cache()
        print("Releasing normal net GPU memory...")

        return normal_images

    @torch.no_grad()
    def predict_segmentation(self, rgb_images):
        sam_checkpoint = 'sam_lib/checkpoints/sam_vit_h_4b8939.pth'
        model_type = 'vit_h'

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device='cuda')

        predictor = SamPredictor(sam)

        # rgb_images
        # - shape: 1 x 3 x h x w
        # - range: [-1.0, 1.0]
        # rgb_images_np
        # - shape: h x w x 3
        # - range: [0, 255]
        rgb_images_np = rgb_images.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
        rgb_images_np = (rgb_images_np * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

        predictor.set_image(rgb_images_np)

        input_point = np.array([[256, 256]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,                
        )

        # delete segment anything model and release GPU memory
        del sam
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

        masks = torch.tensor(masks, dtype=torch.float32, device='cuda')

        return masks
