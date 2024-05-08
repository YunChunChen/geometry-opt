python script/demo.py --sv_cfg config/sv_control_cfg.yaml --mv_cfg config/mv_control_cfg.yaml --prompt "A DSLR photo of a venus. The image has a plain background" --save_dir debug_all_1.0 --mesh_path data/thingi10k_venus_1000.obj --mesh_scale 1.0 --mv_guidance_strength 1.0 --sv_guidance_strength 1.0

python script/demo.py --sv_cfg config/sv_control_cfg.yaml --mv_cfg config/mv_control_cfg.yaml --prompt "A DSLR photo of a venus. The image has a plain background" --save_dir debug_all_0.5 --mesh_path data/thingi10k_venus_1000.obj --mesh_scale 1.0 --mv_guidance_strength 0.0 --sv_guidance_strength 0.5

python script/demo.py --sv_cfg config/sv_control_cfg.yaml --mv_cfg config/mv_control_cfg.yaml --prompt "A DSLR photo of a venus. The image has a plain background" --save_dir debug_all_0.0 --mesh_path data/thingi10k_venus_1000.obj --mesh_scale 1.0 --mv_guidance_strength 0.0 --sv_guidance_strength 0.0
