from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class SVControlConfig:
    resolution: int = 512
    input_mode: str = 'depth'
    num_samples: int = 1
    ddim_steps: int = 20
    guess_mode: bool = False
    strength: float = 1.0
    scale: float = 9.0
    eta: float = 1.0
    seed: int = 12345
    prompt: str = ''
    a_prompt: str = ''
    n_prompt: str = ''

@dataclass
class MVControlConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: str
    pretrained_controlnet_path: str
    revision: Optional[str]
    seed: Optional[int]
    pipe_validation_kwargs: Dict
    pipe_kwargs: Dict
    validation_guidance_scales: List[float]
    unet_from_pretrained_kwargs: Dict
    strength: float = 1.0
    resolution: int = 256

@dataclass
class MeshOptConfig:
    num_iters: int

@dataclass
class Config:
    mesh_opt_cfg: MeshOptConfig
    mv_control_cfg: MVControlConfig
    sv_control_cfg: SVControlConfig
    save_dir: str
    mesh_path: str
    mesh_scale: float
    num_iters: int
