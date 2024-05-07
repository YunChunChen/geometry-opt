import numpy as np
from PIL import Image
from typing import Tuple, List
import os
import math

import torch
from torch.utils.data import Dataset

class Wonder3DTestDataset(Dataset):

    def __init__(self,
        root_dir: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        file_list: List[str],
    ) -> None:

        self.root_dir = root_dir
        self.img_wh = img_wh
        self.bg_color = bg_color
        self.num_views = num_views

        self.views = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        
        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        self.file_list = file_list

    def load_fixed_poses(self):
        fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"
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

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return bg_color
    
    def load_image(self, img_path, bg_color):
        #image = np.array(Image.open(img_path))
        image = np.array(Image.open(img_path).resize(self.img_wh))
        image = image.astype(np.float32) / 255. # [0, 1]

        # split image and mask
        alpha = image[:, :, 3:]
        img   = image[:, :, :3]

        # compare the image
        img = img * alpha + bg_color * (1 - alpha) # [0, 1]
        img = torch.from_numpy(img).permute(2,0,1) # 3 x H x W
        
        return img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        object_name = self.file_list[index]
        image_name = object_name.replace('.png', '')

        # get camera poses
        cond_w2c = self.fix_cam_poses['front']
        tgt_w2cs = [self.fix_cam_poses[view] for view in self.views]

        # get the bg color
        bg_color = self.get_bg_color()

        # read rgb image
        rgb_path = os.path.join(self.root_dir, object_name)
        rgb_in = [self.load_image(rgb_path, bg_color)] * self.num_views
        rgb_in = torch.stack(rgb_in, dim=0).float() # (Nv, 3, H, W)

        elevations, azimuths = [], []
        for view, tgt_w2c in zip(self.views, tgt_w2cs):
            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        # camera embeddings
        elevations = torch.tensor(np.array(elevations)).float().squeeze(1) # Nv
        azimuths = torch.tensor(np.array(azimuths)).float().squeeze(1) # Nv
        elevations_cond = torch.as_tensor([0] * self.num_views).float()
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)

        # normal task embeddings
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)

        # color task embeddings
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)

        data =  {
            'rgb_in': rgb_in,
            'camera_embeddings': camera_embeddings,
            'normal_task_embeddings': normal_task_embeddings,
            'color_task_embeddings': color_task_embeddings,
            'image_name': image_name,
        }

        return data
