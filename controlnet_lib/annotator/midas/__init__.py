# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE

import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference


class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()

    @torch.no_grad()
    def __call__(self, image, normalize=False):
        # forward pass
        depth = self.model(image)
        # depth normalization
        if normalize:
            depth = depth - torch.min(depth)
            depth = depth / torch.max(depth)
        return depth

class MidasDetectorFromRGB:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()

    def __call__(self, input_image, normalize=False):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            if normalize:
                max_depth = torch.max(depth)
                min_depth = torch.min(depth)
                depth = (depth - min_depth) / (max_depth - min_depth)
                depth = depth.cpu().numpy()
                #depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
            else:
                depth = depth.cpu().numpy()
                #depth = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth
