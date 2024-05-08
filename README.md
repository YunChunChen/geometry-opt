# geometry-opt

## Install packages
```
# clone the repo
git clone https://git.azr.adobeitc.com/yunchunc/geometry-opt.git
cd geometry-opt

# conda env
conda create -n geometry-opt -y python=3.9
conda activate geometry-opt

# cuda
pip install -r requirements.txt

# install torch_scatter
See here: https://github.com/rusty1s/pytorch_scatter

# cuda
conda install -c conda-forge cudatoolkit-dev
```

## Install nvdiffrast
```
cd nvdiffrast
pip install .
```

## Download (single-view) ControlNet checkpoints
```
cd ../svcontrol_lib/models
# download ControlNet checkpoints and put them in svcontrol_lib/models
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
```

## Download Segment Anything checkpoint
```
# download segment anything checkpoint and put it in mvcontrol_lib/ckpts/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Download Wonder3D checkpoints
```
# download everything from the link below and put them in mvcontrol_lib/ckpts
Link: https://huggingface.co/flamehaze1115/wonder3d-v1.0/tree/main
```

## Download Multi-view ControlNet checkpoint
```
# download it from the link below, unzip it, and put it in mvcontrol_lib/ckpts
Link: https://drive.google.com/file/d/1EfjtELUOiPr5ANzE-GSV5JOuXfc9_ecv/view?usp=sharing
```

## Folder structure
```
# the mv_control_lib/ckpts folder should look like below

├── ckpts
│   ├── controlnet
│   ├── feature_extractor
│   ├── image_encoder
│   ├── model_index.json
│   ├── sam
│   ├── scheduler
│   ├── unet
│   └── vae
```

## Run in headless mode
```
bash script/demo.sh
```

## Run gradio demo 
```
python gradio_demo_full.py

# then copy paste the link from gradio to your web browser
```
