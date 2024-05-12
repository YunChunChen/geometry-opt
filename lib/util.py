import os
import numpy as np
import einops
import cv2
import imageio
import igl
import matplotlib.pyplot as plt
import random
from PIL import Image
from rembg import remove

import torch
import torch.nn.functional as F
import torchvision

from segment_anything import sam_model_registry, SamPredictor

def read_rgb(path):
    img = imageio.imread(path)
    return img

def save_rgb(image, save_path):
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

def save_image_wonder3d(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return

def gaussian_blur(image):
    blur_func = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=5.0)
    img = blur_func(image)
    return img

def resize_blur(image):
    down_image = F.interpolate(image, scale_factor=1.0/4.0, mode='bilinear', antialias=True)
    up_image = F.interpolate(down_image, scale_factor=4.0, mode='bilinear', antialias=True)
    return up_image

def compose(image, mask, bg_color):
    b, _, h, w = image.size()
    if bg_color == 'black':
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).view(1,-1,1,1).repeat(b,1,h,w).to(image.device)
    elif bg_color == 'white':
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(1,-1,1,1).repeat(b,1,h,w).to(image.device)
    image = image * mask + bg_color * (1 - mask)
    return image

def blur(image, mask, bg_color):
    blur_method = random.choice(['gaussian_blur', 'resize_blur'])
    if blur_method == 'gaussian_blur':
        blur_image = gaussian_blur(image)
    elif blur_method == 'resize_blur':
        blur_image = resize_blur(image)
    b, _, h, w = image.size()
    if bg_color == 'black':
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).view(1,-1,1,1).repeat(b,1,h,w).to(image.device)
    elif bg_color == 'white':
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(1,-1,1,1).repeat(b,1,h,w).to(image.device)
    blur_image = compose(image, mask, bg_color)
    return blur_image

def pred_bbox(image):
    image = Image.fromarray(image.astype(np.uint8))
    image_seg = remove(image)
    alpha = np.asarray(image_seg)[:,:,-1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    bbox = np.array([x_min, y_min, x_max, y_max])
    return bbox

def sam_init(device_id=0):
    sam_checkpoint = 'mvcontrol_lib/ckpts/sam/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'

    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def pred_seg(predictor, image, bbox):
    predictor.set_image(image.astype(np.uint8))
    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)
    mask = torch.tensor(masks_bbox[-1], dtype=torch.float32, device='cuda')
    return mask

def dxdy_normal_image(normal_image: torch.Tensor):
    """
    Computes dx and dy normal derivatives. Returns tuple of tensors.
    Input and outputs are images in the torch format: B x C x H x W
    """

    padded_normal = F.pad(
        normal_image, (1, 1, 1, 1), mode="replicate"
    )

    grad_x_filter = (
        torch.tensor([[[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
        .repeat(3, 1, 1, 1)
        .to(normal_image.device)
    )

    grad_y_filter = (
        torch.tensor([[[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
        .repeat(3, 1, 1, 1)
        .to(normal_image.device)
    )

    dx_normal = F.conv2d(
        padded_normal,
        grad_x_filter,
        padding=0,
        groups=3,
    )

    dy_normal = F.conv2d(
        padded_normal,
        grad_y_filter,
        padding=0,
        groups=3,
    )

    return dx_normal, dy_normal

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

def rgb_to_grayscale(image, normalize):
    grayscale_img = torchvision.transforms.functional.rgb_to_grayscale(image)
    if normalize:
        grayscale_img = grayscale_img / 255.0
    return grayscale_img

def canny_edge(image):
    if len(image.shape) == 4:
        image = (image + 1.0) / 2.0
        image = rgb_to_grayscale(image, normalize=False)
        image = image.squeeze()
    image = (image.data.cpu().numpy() * 255.0).astype(np.uint8)
    edge_image = cv2.Canny(image, 100, 200)
    edge_image = torch.tensor(edge_image, dtype=torch.float32, device='cuda') / 255.0
    return edge_image

def sobel_edge(image, opencv=False, from_rgb=True):
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)

    if from_rgb:
        image = (image + 1.0) / 2.0
        image = rgb_to_grayscale(image, normalize=False)

    # sobel filters
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
        dtype=torch.float32, device='cuda'
    ).unsqueeze(0).unsqueeze(0)
        
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
        dtype=torch.float32, device='cuda'
    ).unsqueeze(0).unsqueeze(0)

    if opencv:
        sobel_x = -1.0 * sobel_x
        sobel_y = -1.0 * sobel_y

    dx = F.conv2d(image, sobel_x, stride=1, padding=1)
    dy = F.conv2d(image, sobel_y, stride=1, padding=1)

    edge_image = torch.sum((dx**2.0 + dy**2.0)**0.5, axis=1).squeeze(0)

    return edge_image
