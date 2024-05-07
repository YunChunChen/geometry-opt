#for obj in dog donut house legoman nascar potion
#for obj in potion
#do
#    python script/controlnet.py --cfg config/single_view/${obj}.yaml --strength 1.0 --save_dir cnet_sv/${obj} --num_iters 2
#done

#python script/controlnet.py --cfg config/single_view/turtle.yaml --strength 0.0 --save_dir debug/turtle --num_iters 1

#python script/controlnet.py --cfg config/single_view/tortoise_top.yaml --strength 0.0 --input_mode depth --save_dir debug/tortoise_depth --num_iters 1

#python script/controlnet.py --cfg config/single_view/dog1.yaml --strength 0.0 --input_mode depth --save_dir exp_pose/dog1 --num_iters 2
#python script/controlnet.py --cfg config/single_view/dog2.yaml --strength 0.0 --input_mode depth --save_dir exp_pose/dog2 --num_iters 2

#python script/controlnet.py --cfg config/single_view/elephant0.yaml --strength 0.0 --input_mode depth --save_dir exp_pose/elephant0 --num_iters 6
#python script/controlnet.py --cfg config/single_view/elephant1.yaml --strength 0.0 --input_mode depth --save_dir exp_pose/elephant1 --num_iters 6
#python script/controlnet.py --cfg config/single_view/elephant2.yaml --strength 0.0 --input_mode depth --save_dir exp_pose/elephant2 --num_iters 6

python script/controlnet.py --cfg config/single_view/thingi10k_cat.yaml --strength 0.0 --input_mode depth --save_dir exp_thingi10k/cat --num_iters 1
python script/controlnet.py --cfg config/single_view/thingi10k_santa.yaml --strength 0.0 --input_mode depth --save_dir exp_thingi10k/santa --num_iters 1
python script/controlnet.py --cfg config/single_view/thingi10k_venus.yaml --strength 0.0 --input_mode depth --save_dir exp_thingi10k/venus --num_iters 1
