import argparse
import os
from typing import Dict, Optional, List
from omegaconf import OmegaConf
from PIL import Image
from dataclasses import dataclass
from packaging import version

import torch

import accelerate
from accelerate.utils import set_seed

import xformers

from mvdiffusion.data.wonder3d_test_dataset import Wonder3DTestDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from utils.misc import load_config    

from tqdm.auto import tqdm
from einops import rearrange
from rembg import remove

weight_dtype = torch.float16

VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    pipe_validation_kwargs: Dict
    validation_guidance_scales: List[float]

def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    ndarr = remove(ndarr)
    im = Image.fromarray(ndarr)
    im.save(fp)

def log_validation(dataloader, pipeline, cfg):

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
    
    for i, batch in tqdm(enumerate(dataloader)):
        # repeat input rgb and reshape
        rgb_in = torch.cat([batch['rgb_in']]*2, dim=0) # 2B x Nv x 3 x H x W
        rgb_in = rearrange(rgb_in, "B Nv C H W -> (B Nv) C H W")

        image_name = batch['image_name']
        
        # embeddings (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)
        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)
        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")

        num_views = len(VIEWS)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    rgb_in, 
                    camera_task_embeddings, 
                    generator=generator, 
                    guidance_scale=guidance_scale, 
                    output_type='pt', 
                    num_images_per_prompt=1, 
                    **cfg.pipe_validation_kwargs
                ).images

                bsz = out.shape[0] // 2
                normals_pred = out[:bsz]
                images_pred = out[bsz:]

                for i in range(bsz//num_views):
                    obj_dir = os.path.join(cfg.save_dir, image_name[i], 'cfg-{}'.format(guidance_scale))
                    os.makedirs(obj_dir, exist_ok=True)
                    for j in range(num_views):
                        view = VIEWS[j]
                        idx = i*num_views + j

                        normal = normals_pred[idx]
                        normal_filename = 'normal_{}.png'.format(view)
                        save_image(normal, os.path.join(obj_dir, normal_filename))

                        color = images_pred[idx]
                        rgb_filename = 'rgb_{}.png'.format(view)
                        save_image(color, os.path.join(obj_dir, rgb_filename))

    torch.cuda.empty_cache()

def main(cfg):

    set_seed(cfg.seed)

    pipeline = MVDiffusionImagePipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    )

    pipeline.unet.enable_xformers_memory_efficient_attention()

    pipeline.to('cuda:0')

    # Get the dataset
    val_dataset = Wonder3DTestDataset(**cfg.validation_dataset)

    # DataLoaders creation:
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.validation_batch_size, 
        shuffle=False, 
        num_workers=cfg.dataloader_num_workers
    )

    os.makedirs(cfg.save_dir, exist_ok=True)

    log_validation(val_dataloader, pipeline, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()

    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    main(cfg)
