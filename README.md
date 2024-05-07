# geometry-opt

## Install packages
```
# clone the repo
git clone https://git.azr.adobeitc.com/yunchunc/geometry-opt.git
cd geometry-opt

# conda env
conda env create -f environment.yaml
conda activate geometry-opt

# cuda
conda install -c conda-forge cudatoolkit-dev
```

## Install nvdiffrast
```
cd nvdiffrast
pip install .
```

## Test nvdiffrast (optional)
```
cd nvdiffrast/samples/torch
python triangle.py --cuda
python cube.py --resolution 16 --outdir cube
python earth.py --outdir earth
python earth.py --mip --outdir earth-mip
python envphong.py --outdir envphong
python pose.py --outdir pose
```

## Download ControlNet checkpoints
```
cd ../controlnet_lib/models
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth
```

## Install Segment Anything
```
pip install git+https://github.com/facebookresearch/segment-anything.git

# download checkpoints to sam_lib/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Test the pipeline
```
# test ControlNet: mesh -> render depth -> ControlNet -> RGB image -> MiDaS -> pred depth
python depth_render_demo.py ../data/sphere.ply
```

## Blender

### Download from the website
```
# download it to the home directory
wget https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.1-linux-x64.tar.xz

# unzip the file
tar Jxvf blender-3.6.1-linux-x64.tar.xz

# in bashrc, add
alias blender="~/blender-3.6.1-linux-x64/blender"
```

### Copy from sensei-fs
```
# copy it to the home directory
cp -r /sensei-fs/users/yunchunc/blender-3.6.1-linux-x64 ~/.

# in bashrc, add
alias blender="~/blender-3.6.1-linux-x64/blender"
```
