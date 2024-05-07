import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Any
import random
import json
import os
import math

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision

from .normal_utils import trans_normal, img2normal

class MVControlNetTestDataset(Dataset):

    def __init__(self,
        root_dir: str,
        num_views: int,
        bg_color: Any,
        img_wh: Tuple[int, int],
        object_list: str,
        invalid_list: Optional[str] = None,
        validation: bool = False,
        control_type: str = 'normal',
        num_validation_samples: int = 64
    ) -> None:

        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.bg_color = bg_color
        self.img_wh = img_wh
        self.validation = validation
        self.control_type = control_type
        self.invalid_list = invalid_list

        self.views  = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        # data list
        with open(object_list) as f:
            all_objects = json.load(f)

        with open(self.invalid_list) as f:
            invalid_objects = json.load(f)
        
        all_objects = set(all_objects) - (set(invalid_objects) & set(all_objects))
        all_objects = list(all_objects)

        if not validation:
            self.all_objects = all_objects[:-num_validation_samples]
        else:
            self.all_objects = all_objects[-num_validation_samples:]

        print("Loading ", len(self.all_objects), " objects in the dataset")

    def __len__(self):
        return len(self.all_objects) * self.total_view

    def load_fixed_poses(self):
        fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"
        poses = {}
        for face in self.views:
            RT = np.loadtxt(os.path.join(fix_cam_pose_dir,'000_{}_RT.txt'.format(face)))
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

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        return bg_color

    def load_image(self, img_path, bg_color):
        image = np.array(Image.open(img_path).resize(self.img_wh))
        image = image.astype(np.float32) / 255. # [0, 1]

        # split image and mask
        alpha = image[:, :, 3:]
        img   = image[:, :, :3]

        # compose the image
        img = img * alpha + bg_color * (1 - alpha) # [0, 1]
        img = torch.from_numpy(img).permute(2,0,1) # 3 x H x W

        return img
    
    def load_normal(self, img_path, bg_color, RT_w2c=None, RT_w2c_cond=None):
        image = np.array(Image.open(img_path).resize(self.img_wh))

        # split image and mask
        alpha  = image[:, :, 3:] / 255.0
        normal = image[:, :, :3] / 255.0

        # convert coordinate frame
        #normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond) # [-1, 1]

        img = (normal).astype(np.float32)  # [0, 1]

        img = img * 2.0 - 1.0

        img[:,:,0] *= -1.0

        #img = (normal*0.5 + 0.5).astype(np.float32)  # [0, 1]

        # compose the image
        img = img * alpha + bg_color * (1 - alpha) # [0, 1]
        img = torch.from_numpy(img).permute(2,0,1) # 3 x H x W

        alpha = torch.from_numpy(alpha).permute(2,0,1) # 1 x H x W
        
        #return img, alpha
        return torch.from_numpy(image).permute(2,0,1)[:3,:,:], alpha

    def gaussian_blur(self, image):
        blur_func = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=5.0)
        img = blur_func(image)
        return img

    def resize_blur(self, image):
        down_image = F.interpolate(image, scale_factor=1.0/4.0, mode='bilinear', antialias=True)
        up_image = F.interpolate(down_image, scale_factor=4.0, mode='bilinear', antialias=True)
        return up_image

    def blur(self, image, mask, bg_color):
        blur_method = random.choice(['gaussian_blur', 'resize_blur'])
        if blur_method == 'gaussian_blur':
            blur_image = self.gaussian_blur(image)
        elif blur_method == 'resize_blur':
            blur_image = self.resize_blur(image)
        b, _, h, w = image.size()
        bg_color = torch.from_numpy(bg_color).view(1,-1,1,1).repeat(b,1,h,w)
        blur_image = blur_image * mask + bg_color * (1 - mask)
        return blur_image

    def __len__(self):
        return len(self.all_objects)

    def __getitem__(self, index):
        object_name = self.all_objects[index % len(self.all_objects)]

        # get camera poses
        cond_w2c = self.fix_cam_poses['front']
        tgt_w2cs = [self.fix_cam_poses[view] for view in self.views]

        # get the bg color
        bg_color = self.get_bg_color()

        # read input RGB 
        #rgb_path = os.path.join(self.root_dir, object_name, 'rgb_front.png')
        rgb_path = os.path.join(self.root_dir, object_name, 'rgb_0.png')
        rgb_in = [self.load_image(rgb_path, bg_color)] * self.num_views

        rgb_out, normal_out, mask_out = [], [], []
        elevations, azimuths = [], []
        idx = 0
        for view, tgt_w2c in zip(self.views, tgt_w2cs):
            
            # read output RGB
            #rgb_path = os.path.join(self.root_dir, object_name, 'rgb_{}.png'.format(view))
            #rgb_tensor = self.load_image(rgb_path, bg_color)
            #rgb_out.append(rgb_tensor)

            # read output normal and mask
            #normal_path = os.path.join(self.root_dir, object_name, 'normals_{}.png'.format(view))
            normal_path = os.path.join(self.root_dir, object_name, 'normal_{}.png'.format(idx))
            normal_tensor, mask_tensor = self.load_normal(normal_path, bg_color, RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c)
            normal_out.append(normal_tensor)
            mask_out.append(mask_tensor)

            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

            idx += 1

        # input and output RGB
        rgb_in  = torch.stack(rgb_in, dim=0).float() # (Nv, 3, H, W)
        #rgb_out = torch.stack(rgb_out, dim=0).float() # (Nv, 3, H, W)

        # output mask
        mask_out = torch.stack(mask_out, dim=0).float() # (Nv, 1, H, W)

        # input and output normal
        normal_out = torch.stack(normal_out, dim=0).float() # (Nv, 3, H, W)
        normal_in  = self.blur(normal_out, mask_out, bg_color) # (Nv, 3, H, W)

        # camera embeddings
        elevations = torch.tensor(np.array(elevations)).float().squeeze(1) # Nv
        azimuths = torch.tensor(np.array(azimuths)).float().squeeze(1) # Nv
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 6 views to train
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)

        # normal task embeddings
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)

        # color task embeddings
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)

        data = {
            'rgb_in': rgb_in,
            #'rgb_out': rgb_out,
            #'normal_in': normal_in,
            'normal_in': normal_out,
            #'normal_out': normal_out,
            'camera_embeddings': camera_embeddings,
            'normal_task_embeddings': normal_task_embeddings,
            'color_task_embeddings': color_task_embeddings,
        }

        return data
