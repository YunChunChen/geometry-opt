from yacs.config import CfgNode as CN

_C = CN()

# Experiment configs
_C.exp = CN()
_C.exp.save_dir = ''
_C.exp.num_iters = 1000
_C.exp.save_freq = 100   # save files every 100 iterations
_C.exp.remeshing = False
_C.exp.remesh_freq = 200 # remeshing every 200 iterations

# Data configs
_C.data = CN()
_C.data.mesh_path = ''
_C.data.mesh_scale = 1.0 # mesh scaling factor
_C.data.num_subdiv = 0   # number of subdivisions
_C.data.gt_from = ''     # ground truth from [controlnet | some folder]

# Optimizer configs
_C.optim = CN()
_C.optim.adam_lr = 1e-3
_C.optim.largesteps_lr = 1e-3
_C.optim.lambda_ = 10    # lambda for the largestep optimizer

# Renderer configs
_C.renderer = CN()
_C.renderer.resolution = 512

# ControlNet configs
_C.controlnet = CN()
_C.controlnet.from_depth = False  # controlnet input condition
_C.controlnet.from_normal = False # controlnet input condition
_C.controlnet.pred_depth = False  # controlnet output analysis
_C.controlnet.pred_normal = False # controlnet output analysis
_C.controlnet.num_samples = 1
_C.controlnet.resolution = 512
_C.controlnet.ddim_steps = 20
_C.controlnet.guess_mode = False
_C.controlnet.strength = 1.0
_C.controlnet.scale = 9.0
_C.controlnet.eta = 1.0
_C.controlnet.seed = 12345
_C.controlnet.prompt = ''
_C.controlnet.a_prompt = ''
_C.controlnet.n_prompt = ''

def get_cfg_defaults():
    return _C.clone()
