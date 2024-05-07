#accelerate launch --config_file 2gpu.yaml train_controlwonder3d.py --config configs/train/train.yaml

#accelerate launch --config_file 4gpu.yaml train_controlwonder3d.py --config configs/train/train.yaml

#accelerate launch --config_file 4gpu.yaml train_wonder3d.py --config configs/train/train.yaml

accelerate launch --config_file 4gpu.yaml train_controlnet.py --config configs/train/train.yaml

#accelerate launch --config_file 2gpu.yaml train_controlnet.py --config configs/train/train.yaml
