IMG_DIR=cnet_sv

for obj in dog donut house legoman nascar potion
do
    for id in 000 001
    do
        python script/rm_background.py ${IMG_DIR}/${obj}/${id}-rgb.png ${IMG_DIR}/${obj}/${id}-rgba.png
    done
done
