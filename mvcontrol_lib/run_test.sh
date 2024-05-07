#accelerate launch --config_file 1gpu.yaml test_wonder3d.py --config configs/test_wonder3d.yaml

accelerate launch --config_file 1gpu.yaml test_mvcontrolnet.py --config configs/test_mvcontrolnet.yaml
