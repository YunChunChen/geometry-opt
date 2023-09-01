import os
import numpy as np
import einops
import cv2
import imageio
import igl
import matplotlib.pyplot as plt

def save_rgb(image, save_path):
    # can be used for saving RGB and normal
    image_np = (
        einops.rearrange(image, 'c h w -> h w c') * 127.5 + 127.5
    ).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    imageio.imsave(save_path, image_np)
    return

def save_depth(depth, save_path, cmap):
    depth_np = depth.detach().cpu().numpy()
    if cmap:
        plt.imshow(depth_np, cmap='viridis', vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        depth_np = np.clip(np.rint(depth_np*255.0), 0, 255).astype(np.uint8)
        imageio.imsave(save_path, depth_np)
    return

def save_seg(seg, save_path):
    seg_np = seg.detach().cpu().numpy()
    seg_np = np.clip(np.rint(seg_np * 255.0), 0, 255).astype(np.uint8)
    imageio.imsave(save_path, seg_np)
    return

def save_npz(data, save_path):
    data_np = data.detach().cpu().numpy()
    np.savez(save_path, data=data_np)
    return

def save_mesh(mesh, save_path):
    mesh_v = mesh.get_vertices().data.cpu().numpy()
    mesh_f = mesh.get_faces().data.cpu().numpy()
    igl.write_triangle_mesh(save_path, mesh_v, mesh_f)
    return

def image_to_video(image_dir, save_path):
    num_images = len(os.listdir(image_dir))
    print("Num of images:", num_images)
    images = ['{}/{:06d}.png'.format(image_dir, i) for i in range(num_images)]
    frame = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    height, width = frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, 50.0, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    return
