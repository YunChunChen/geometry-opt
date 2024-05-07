# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings
from typing import Callable, List, Optional, Union

import PIL
import torch
import torchvision.transforms.functional as TF
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from einops import rearrange, repeat

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MVControlNetImagePipeline(DiffusionPipeline):
    # TODO: feature_extractor is required to encode images (if they are in PIL format),
    # we should give a descriptive message if the pipeline doesn't have one.
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        camera_embedding_type: str = 'e_de_da_sincos',
        num_views: int = 6
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            controlnet=controlnet, # new
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor) # new

        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.camera_embedding_type: str = camera_embedding_type
        self.num_views: int = num_views

        self.camera_embedding =  torch.tensor(
            [[ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000],
            [ 0.0000, -0.2362,  0.8125,  1.0000,  0.0000],
            [ 0.0000, -0.1686,  1.6934,  1.0000,  0.0000],
            [ 0.0000,  0.5220,  3.1406,  1.0000,  0.0000],
            [ 0.0000,  0.6904,  4.8359,  1.0000,  0.0000],
            [ 0.0000,  0.3733,  5.5859,  1.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
            [ 0.0000, -0.2362,  0.8125,  0.0000,  1.0000],
            [ 0.0000, -0.1686,  1.6934,  0.0000,  1.0000],
            [ 0.0000,  0.5220,  3.1406,  0.0000,  1.0000],
            [ 0.0000,  0.6904,  4.8359,  0.0000,  1.0000],
            [ 0.0000,  0.3733,  5.5859,  0.0000,  1.0000]], dtype=torch.float16)

    def _encode_image(self, image_pil, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        image_pt = self.feature_extractor(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image_pt).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        # Note: repeat differently from official pipelines
        # B1B2B3B4 -> B1B2B3B4B1B2B3B4
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(num_images_per_prompt, 1, 1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
        
        image_pt = torch.stack([TF.to_tensor(img) for img in image_pil], dim=0).to(device).to(dtype)
        image_pt = image_pt * 2.0 - 1.0
        image_latents = self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor
        # Note: repeat differently from official pipelines
        # B1B2B3B4 -> B1B2B3B4B1B2B3B4        
        image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            image_latents = torch.cat([torch.zeros_like(image_latents), image_latents])

        return image_embeddings, image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance=False, guess_mode=False):
        #image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        print('control image dtype:', image.dtype)
        print('control image shape:', image.shape)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_camera_embedding(self, camera_embedding: Union[float, torch.Tensor], do_classifier_free_guidance, num_images_per_prompt=1):
        # (B, 3)
        camera_embedding = camera_embedding.to(dtype=self.unet.dtype, device=self.unet.device)

        if self.camera_embedding_type == 'e_de_da_sincos':
            # (B, 6)
            camera_embedding = torch.cat([
                torch.sin(camera_embedding),
                torch.cos(camera_embedding)
            ], dim=-1)
            assert self.unet.config.class_embed_type == 'projection'
            assert self.unet.config.projection_class_embeddings_input_dim == 6 or self.unet.config.projection_class_embeddings_input_dim == 10
        else:
            raise NotImplementedError
        
        # Note: repeat differently from official pipelines
        # B1B2B3B4 -> B1B2B3B4B1B2B3B4        
        camera_embedding = camera_embedding.repeat(num_images_per_prompt, 1)     

        if do_classifier_free_guidance:
            camera_embedding = torch.cat([camera_embedding, camera_embedding], dim=0)
        
        return camera_embedding    

    @torch.no_grad()
    def __call__(
        self,
        image: Union[List[PIL.Image.Image], torch.FloatTensor],
        camera_embedding: Optional[torch.FloatTensor]=None,
        control_image: Optional[Union[List[PIL.Image.Image], torch.FloatTensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width, callback_steps)

        # 2. Define call parameters
        if isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
            assert batch_size >= self.num_views and batch_size % self.num_views == 0
        elif isinstance(image, PIL.Image.Image):
            image = [image]*self.num_views*2
            batch_size = self.num_views*2
    
        device = self._execution_device
        dtype = self.vae.dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale != 1.0

        # 3. Encode input image
        if isinstance(image, list):
            image_pil = image
        elif isinstance(image, torch.Tensor):
            image_pil = [TF.to_pil_image(image[i]) for i in range(image.shape[0])]
        image_embeddings, image_latents = self._encode_image(image_pil, device, num_images_per_prompt, do_classifier_free_guidance)

        # prepare control image
        #control_image = self.prepare_control_image(
        #    image=control_image,
        #    width=width,
        #    height=height,
        #    batch_size=batch_size * num_images_per_prompt,
        #    num_images_per_prompt=num_images_per_prompt,
        #    device=device,
        #    dtype=controlnet.dtype,
        #    do_classifier_free_guidance=do_classifier_free_guidance,
        #)

        if do_classifier_free_guidance:
            control_image = torch.cat([control_image]*2).to(device).to(self.controlnet.dtype)
        if camera_embedding is not None:
            assert len(camera_embedding) == batch_size
        else:
            camera_embedding = self.camera_embedding.to(dtype)
            camera_embedding = repeat(camera_embedding, "Nv Nce -> (B Nv) Nce", B=batch_size//len(camera_embedding))
        camera_embeddings = self.prepare_camera_embedding(camera_embedding, do_classifier_free_guidance=do_classifier_free_guidance, num_images_per_prompt=num_images_per_prompt)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.out_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet inference
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input, 
                    t,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_embeddings,
                    controlnet_cond=control_image,
                    return_dict=False
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=image_embeddings, 
                    class_labels=camera_embeddings,
                    down_block_additional_residuals=[sample.to(dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype)
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            if num_channels_latents == 8:
                latents = torch.cat([latents[:, :4], latents[:, 4:]], dim=0)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, image_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
