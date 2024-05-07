import argparse
import logging
import math
import os
from typing import Dict, Optional, List
from omegaconf import OmegaConf
import numpy as np
from dataclasses import dataclass
from packaging import version

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mvdiffusion.data.mv_dataset import MVDataset
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

import wandb
from einops import rearrange

from tqdm.auto import tqdm
import time

logger = get_logger(__name__, log_level='INFO')

@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    revision: Optional[str]
    train_dataset: Dict
    validation_dataset: Dict
    output_dir: str
    seed: Optional[int]
    train_batch_size: int
    validation_batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    snr_gamma: Optional[float]
    use_ema: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    logging_dir: str
    vis_dir: str
    mixed_precision: Optional[str]
    report_to: Optional[str]
    local_rank: int
    checkpointing_steps: int
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_sanity_check: bool
    tracker_project_name: str
    exp_name: str
    wandb_dir: str
    num_views: int
    last_global_step: int
    trainable_modules: Optional[list]
    use_classifier_free_guidance: bool
    condition_drop_rate: float
    scale_input_latents: bool
    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    camera_embedding_lr_mult: float

def log_vis(
    cfg, 
    accelerator,
    input_img, 
    pred_img, 
    gt_img, 
    name, 
    global_step
):
    input_img = torch.cat(input_img, dim=0)

    pred_img = torch.cat(pred_img, dim=0)
    pred_img = rearrange(pred_img, "(B Nv) C H W -> B Nv C H W", Nv=cfg.num_views)

    gt_img = torch.cat(gt_img, dim=0)

    combine_img = torch.stack((input_img, pred_img, gt_img), dim=1)
    combine_img = rearrange(combine_img, "B K Nv C H W -> (B K Nv) C H W")

    nrow = cfg.num_views
    ncol = combine_img.shape[0] // nrow
    grid_img = make_grid(combine_img, nrow=nrow, ncol=ncol, padding=0, value_range=(0, 1))
    grid_img = wandb.Image(grid_img, caption="{} Input | Pred | GT".format(name))
    accelerator.log({"{} Input | Pred | GT".format(name): grid_img}, step=global_step)

def log_validation(
    dataloader, 
    vae, 
    feature_extractor, 
    image_encoder, 
    unet, 
    cfg: TrainingConfig, 
    accelerator, 
    weight_dtype, 
    global_step, 
    name, 
    save_dir
):
    logger.info(f"Running {name} ... ")

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder, 
        feature_extractor=feature_extractor, 
        vae=vae, 
        unet=accelerator.unwrap_model(unet), 
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='scheduler'),
        **cfg.pipe_kwargs
    )

    pipeline.set_progress_bar_config(disable=True)

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()    

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    rgb_in_all, rgb_gt_all, rgb_pred_all = [], [], []
    normal_in_all, normal_gt_all, normal_pred_all = [], [], []
    for i, batch in enumerate(dataloader):
        # get data
        rgb_in     = batch['rgb_in']     # B x Nv x 3 x H x W
        rgb_out    = batch['rgb_out']    # B x Nv x 3 x H x W
        normal_in  = batch['normal_in']  # B x Nv x 3 x H x W
        normal_out = batch['normal_out'] # B x Nv x 3 x H x W

        # append rgb and normal input and ground truth
        rgb_in_all.append(rgb_in)
        rgb_gt_all.append(rgb_out)
        normal_in_all.append(normal_in)
        normal_gt_all.append(normal_out)

        # repeat input rgb and reshape
        rgb_in = torch.cat([rgb_in]*2, dim=0) # 2B x Nv x 3 x H x W
        rgb_in = rearrange(rgb_in, "B Nv C H W -> (B Nv) C H W")
        
        # embeddings (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)
        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)
        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")

        with torch.autocast('cuda'):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                start_time = time.time()
                out = pipeline(
                    rgb_in, 
                    camera_task_embeddings, 
                    generator=generator, 
                    guidance_scale=guidance_scale, 
                    output_type='pt', 
                    num_images_per_prompt=1, 
                    **cfg.pipe_validation_kwargs
                ).images
                end_time = time.time()
                print("eval time:", end_time - start_time)

                shape = out.shape
                out0, out1 = out[:shape[0]//2], out[shape[0]//2:]
                normal_pred = []
                rgb_pred = []
                for ii in range(shape[0]//2):
                    normal_pred.append(out0[ii])
                    rgb_pred.append(out1[ii])

                normal_pred = torch.stack(normal_pred, dim=0)
                normal_pred_all.append(normal_pred)

                rgb_pred = torch.stack(rgb_pred, dim=0)
                rgb_pred_all.append(rgb_pred)

    log_vis(cfg, accelerator, rgb_in_all, rgb_pred_all, rgb_gt_all, 'RGB', global_step)
    log_vis(cfg, accelerator, normal_in_all, normal_pred_all, normal_gt_all, 'Normal', global_step)

    torch.cuda.empty_cache()

def main(cfg):
    # override local_rank with envvar
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank

    vis_dir = os.path.join(cfg.output_dir, cfg.vis_dir)
    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(cfg.seed)

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='scheduler')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='image_encoder', revision=cfg.revision)
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='feature_extractor', revision=cfg.revision)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='vae', revision=cfg.revision)
    unet = UNetMV2DConditionModel.from_pretrained(cfg.pretrained_unet_path, subfolder='unet', revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    print("Loading pre-trained unet from", cfg.pretrained_unet_path)

    if cfg.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNetMV2DConditionModel, model_config=unet.config)

    def compute_snr(timesteps):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    if cfg.trainable_modules is None:
        unet.requires_grad_(True)
    else:
        unet.requires_grad_(False)
        for name, module in unet.named_modules():
            if name.endswith(tuple(cfg.trainable_modules)):
                for params in module.parameters():
                    params.requires_grad = True                

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warning("If observe problems during training, update xFormers to at least 0.0.17.")
            unet.enable_xformers_memory_efficient_attention()
            print("use xformers to speed up")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, 'unet_ema'))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, 'unet'))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, 'unet_ema'), UNetMV2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetMV2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    torch.backends.cuda.matmul.allow_tf32 = True        

    params, params_class_embedding = [], []
    for name, param in unet.named_parameters():
        if 'class_embedding' in name:
            params_class_embedding.append(param)
        else:
            params.append(param)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [
            {'params': params, 'lr': cfg.learning_rate},
            {'params': params_class_embedding, 'lr': cfg.learning_rate * cfg.camera_embedding_lr_mult}
        ],
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )

    # Get the training dataset
    train_dataset = MVDataset(**cfg.train_dataset)
    val_dataset = MVDataset(**cfg.validation_dataset)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train_batch_size, 
        shuffle=True, 
        num_workers=cfg.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.validation_batch_size, 
        #shuffle=False, 
        shuffle=True, 
        num_workers=cfg.dataloader_num_workers
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    if cfg.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16

    # Move text_encode and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        os.makedirs(cfg.wandb_dir, exist_ok=True)
        accelerator.init_trackers(
            project_name=cfg.tracker_project_name, 
            config={},
            init_kwargs={'wandb': {'name': cfg.exp_name, 'dir': cfg.wandb_dir}}
        )    

    # Train
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != 'latest':
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.exists(os.path.join(cfg.output_dir, 'checkpoint')):
                path = 'checkpoint'
            else:
                dirs = os.listdir(cfg.output_dir)
                dirs = [d for d in dirs if d.startswith('checkpoint')]
                dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run.")
            cfg.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = cfg.last_global_step

            resume_global_step = global_step * cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.gradient_accumulation_steps)        

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Steps')

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfg.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfg.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # (B, Nv, 3, H, W)
                imgs_in, colors_out, normals_out = batch['rgb_in'], batch['rgb_out'], batch['normal_out']

                rgb_in     = batch['rgb_in']     # B x Nv x 3 x H x W
                rgb_out    = batch['rgb_out']    # B x Nv x 3 x H x W
                normal_in  = batch['normal_in']  # B x Nv x 3 x N x W
                normal_out = batch['normal_out'] # B x Nv x 3 x H x W

                B, Nv = rgb_in.shape[:2]
                
                # repeat  (2B, Nv, 3, H, W)
                rgb_in  = torch.cat([rgb_in]*2, dim=0)
                rgb_out = torch.cat([normal_out, rgb_out], dim=0)

                # reshape 
                rgb_in  = rearrange(rgb_in, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
                rgb_out = rearrange(rgb_out, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
                
                # embeddings (2B, Nv, Nce)
                camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)
                task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)
                camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)
                camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")

                # positional encoding (B*Nv, Nce')
                camera_task_embeddings = torch.cat([torch.sin(camera_task_embeddings), torch.cos(camera_task_embeddings)], dim=-1).to(weight_dtype)

                # (B*Nv, 4, Hl, Wl)
                cond_vae_embeddings = vae.encode(rgb_in * 2.0 - 1.0).latent_dist.mode()
                if cfg.scale_input_latents:
                    cond_vae_embeddings = cond_vae_embeddings * vae.config.scaling_factor
                latents = vae.encode(rgb_out * 2.0 - 1.0).latent_dist.sample() * vae.config.scaling_factor

                rgb_in_proc = TF.resize(rgb_in, (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC)
                # do the normalization in float32 to preserve precision
                rgb_in_proc = ((rgb_in_proc.float() - clip_image_mean) / clip_image_std).to(weight_dtype)        

                # (B*Nv, 1, 768)
                image_embeddings = image_encoder(rgb_in_proc).image_embeds.unsqueeze(1)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # same noise for different views of the same object
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz // cfg.num_views,), device=latents.device).repeat_interleave(cfg.num_views)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Conditioning dropout to support classifier-free guidance during inference. 
                if cfg.use_classifier_free_guidance and cfg.condition_drop_rate > 0.:
                    # drop a group of normals and colors as a whole
                    random_p = torch.rand(B, device=latents.device, generator=generator)
                        
                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - ((random_p >= cfg.condition_drop_rate).to(image_mask_dtype) * (random_p < 3 * cfg.condition_drop_rate).to(image_mask_dtype))
                    image_mask = image_mask.reshape(B, 1, 1, 1, 1).repeat(1, Nv, 1, 1, 1)
                    image_mask = rearrange(image_mask, "B Nv C H W -> (B Nv) C H W")
                    image_mask = torch.cat([image_mask]*2, dim=0)

                    # Final image conditioning.
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - ((random_p < 2 * cfg.condition_drop_rate).to(clip_mask_dtype))
                    clip_mask = clip_mask.reshape(B, 1, 1, 1).repeat(1, Nv, 1, 1)
                    clip_mask = rearrange(clip_mask, "B Nv M C -> (B Nv) M C")
                    clip_mask = torch.cat([clip_mask]*2, dim=0)

                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings
                
                # (B*Nv, 8, Hl, Wl)
                latent_model_input = torch.cat([noisy_latents, cond_vae_embeddings], dim=1)

                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_task_embeddings
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}") 

                if cfg.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()                    

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({'train_loss': train_loss}, step=global_step)
                
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfg.output_dir, 'checkpoint')
                        accelerator.save_state(save_path)
                        try:
                            unet.module.save_pretrained(os.path.join(cfg.output_dir, f'unet-{global_step}/unet'))
                        except:
                            unet.save_pretrained(os.path.join(cfg.output_dir, f'unet-{global_step}/unet'))
                        logger.info(f"Saved state to {save_path}")

                if global_step % cfg.validation_steps == 0 or (cfg.validation_sanity_check and global_step == 1):
                    if accelerator.is_main_process:
                        if cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
    
                        val_dataloader = accelerator.prepare(val_dataloader)

                        log_validation(
                            val_dataloader,
                            vae,
                            feature_extractor,
                            image_encoder,
                            unet,
                            cfg,
                            accelerator,
                            weight_dtype,
                            global_step,
                            'validation',
                            vis_dir
                        )

                        if cfg.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())                        

            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if cfg.use_ema:
            ema_unet.copy_to(unet.parameters())
        pipeline = MVDiffusionImagePipeline(
            image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=unet, safety_checker=None,
            scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder='scheduler'),
            **cfg.pipe_kwargs
        )            
        os.makedirs(os.path.join(cfg.output_dir, 'pipeckpts'), exist_ok=True)
        pipeline.save_pretrained(os.path.join(cfg.output_dir, 'pipeckpts'))

    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
