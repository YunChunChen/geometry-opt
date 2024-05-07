import os
import sys

# add controlnet_repo to sys.path
svcontrol_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'svcontrol_lib'
)
sys.path.append(svcontrol_dir)

import gc
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

import svcontrol_lib.config
from svcontrol_lib.share import *
from svcontrol_lib.cldm.model import create_model, load_state_dict
from svcontrol_lib.cldm.ddim_hacked import DDIMSampler

#from svcontrol_lib.annotator.midas import MidasDetector
#from svcontrol_lib.annotator.normalbae import NormalBaeDetector

#from omnidata_lib.normal import NormalOmniData

#from dsine_lib.models.dsine import DSINE
#import dsine_lib.utils.utils as utils

#from segment_anything import sam_model_registry, SamPredictor

class SVControlNet:
    
    def __init__(self, cfg, guidance=1.0):
        self.num_samples = cfg.num_samples
        self.ddim_steps = cfg.ddim_steps
        self.eta = cfg.eta
        self.scale = cfg.scale
        self.guess_mode = cfg.guess_mode
        self.strength = cfg.strength
        self.input_mode = cfg.input_mode
        self.prompt = cfg.prompt
        self.a_prompt = cfg.a_prompt
        self.n_prompt = cfg.n_prompt
        self.guidance = guidance

    @torch.no_grad()
    def get_depth_network(self):
        model = create_model(
            'svcontrol_lib/models/control_v11f1p_sd15_depth.yaml'
        ).cpu()
        model.load_state_dict(
            load_state_dict('svcontrol_lib/models/v1-5-pruned.ckpt', location='cuda'), 
            strict=False
        )
        model.load_state_dict(
            load_state_dict('svcontrol_lib/models/control_v11f1p_sd15_depth.pth', location='cuda'), 
            strict=False
        )
        model = model.cuda()
        return model

    @torch.no_grad()
    def get_normal_network(self):
        model = create_model(
            'svcontrol_lib/models/control_v11p_sd15_normalbae.yaml'
        ).cpu()
        model.load_state_dict(
            load_state_dict('svcontrol_lib/models/v1-5-pruned.ckpt', location='cuda'), 
            strict=False
        )
        model.load_state_dict(
            load_state_dict('svcontrol_lib/models/control_v11p_sd15_normalbae.pth', location='cuda'), 
            strict=False
        )
        model = model.cuda()
        return model

    @torch.no_grad()
    def get_canny_network(self):
        model = create_model(
            'svcontrol_lib/models/control_v11p_sd15_canny.yaml'
        ).cpu()
        model.load_state_dict(
            load_state_dict('svcontrol_lib/models/v1-5-pruned.ckpt', location='cuda'), 
            strict=False
        )
        model.load_state_dict(
            load_state_dict('svcontrol_lib/models/control_v11p_sd15_canny.pth', location='cuda'), 
            strict=False
        )
        model = model.cuda()
        return model

    @torch.no_grad()
    def get_ddim_sampler(self, model):
        ddim_sampler = DDIMSampler(model)
        return ddim_sampler

    @torch.no_grad()
    def generate_rgb(self, geom_image):

        if self.input_mode == 'depth':
            controlnet = self.get_depth_network()
            geom_image = geom_image.unsqueeze(0)#.unsqueeze(0) # h x w -> 1 x 1 x h x w
            geom_image = geom_image.repeat(self.num_samples, 3, 1, 1)
        elif self.input_mode == 'normal':
            controlnet = self.get_normal_network()
            geom_image = geom_image.unsqueeze(0) # 3 x h x w -> 1 x 3 x h x w
            geom_image = geom_image.repeat(self.num_samples, 1, 1, 1)
        elif self.input_mode == 'canny':
            controlnet = self.get_canny_network()
            geom_image = geom_image.unsqueeze(0).unsqueeze(0) # h x w -> 1 x 1 x h x w
            geom_image = geom_image.repeat(self.num_samples, 3, 1, 1)

        ddim_sampler = self.get_ddim_sampler(controlnet)

        if svcontrol_lib.config.save_memory:
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

        if svcontrol_lib.config.save_memory:
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
            self.guidance,
            cond, 
            verbose=False, 
            eta=self.eta,
            unconditional_guidance_scale=self.scale,
            unconditional_conditioning=un_cond
        )

        if svcontrol_lib.config.save_memory:
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
    def predict_normal_bae(self, rgb_images):
        normal_net = NormalBaeDetector()
        rgb_images = (rgb_images + 1.0) / 2.0
        normal_images = normal_net(rgb_images)

        # delete normal network and release GPU memory
        del normal_net
        gc.collect()
        torch.cuda.empty_cache()
        print("Releasing normal bae GPU memory...")

        return normal_images

    @torch.no_grad()
    def predict_normal_omnidata(self, rgb_images):
        normal_net = NormalOmniData()
        rgb_images = (rgb_images + 1.0) / 2.0
        normal_images = normal_net(rgb_images)

        # delete normal network and release GPU memory
        del normal_net
        gc.collect()
        torch.cuda.empty_cache()
        print("Releasing omnidata normal GPU memory...")

        return normal_images

    @torch.no_grad()
    def predict_normal_dsine(self, rgb_images):
        
        # [-1, 1] to [0, 1]
        rgb_images = (rgb_images + 1.0) / 2.0

        _, _, h, w = rgb_images.shape

        # zero-padding
        l, r, t, b = utils.pad_input(h, w)
        rgb_images = F.pad(rgb_images, (l, r, t, b), mode='constant', value=0.0)

        # normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_images = normalize(rgb_images)

        # get intrinsics
        device = torch.device('cuda')
        intrins = utils.get_intrins_from_fov(new_fov=60.0, H=h, W=w, device=device).unsqueeze(0)
        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        # normal networ
        normal_net = DSINE().cuda()
        normal_net.pixel_coords = normal_net.pixel_coords.cuda()
        normal_net = utils.load_checkpoint('dsine_lib/checkpoints/dsine.pt', normal_net)
        normal_net.eval()

        # prediction
        normal_images = normal_net(rgb_images, intrins=intrins)[-1]
        normal_images = normal_images[:, :, t:t+h, l:l+w]

        # delete normal network and release GPU memory
        del normal_net
        gc.collect()
        torch.cuda.empty_cache()
        print("Releasing dsine normal GPU memory...")

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
