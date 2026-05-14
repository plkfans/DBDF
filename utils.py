import cv2
import numpy as np
from args import args_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches

args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size

def KD_loss(outputs, targets):
	kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/args.temperature, dim=1),F.softmax(targets/args.temperature, dim=1)) * args.temperature * args.temperature
	return kd_loss

def generate_local_images(
    data,
    att,
    out_size=224,
    eps=1e-8,
    roi_thr=0.3,
    roi_boost=1.5,
    smooth_kernel=7,
    return_bbox=False
):
    B, C, H, W = data.shape
    device = data.device

    bboxes = []

    # 1. 通道平均 + 上采样到原图
    att = att.mean(dim=1, keepdim=True)
    att = F.interpolate(att, size=(H, W), mode='bilinear', align_corners=False)

    att_min = att.amin(dim=(-2,-1), keepdim=True)
    att_max = att.amax(dim=(-2,-1), keepdim=True)
    att = (att - att_min) / (att_max - att_min + eps)

    # 2. 自动 ROI（基于阈值）
    att_flat = att.view(B, -1)
    thr = roi_thr * att_flat.max(dim=1, keepdim=True)[0]
    masks = (att_flat > thr).view(B, H, W)
    att_new = att.clone()

    for b in range(B):
        ys, xs = torch.where(masks[b])
        if len(xs) == 0:
            bboxes.append(None)
            continue

        x1 = xs.min().item()
        x2 = xs.max().item()
        y1 = ys.min().item()
        y2 = ys.max().item()

        bboxes.append((x1, y1, x2, y2))

        high_value = att[b].max() * roi_boost
        att_new[b, :, y1:y2, x1:x2] = high_value

    # 3. 边缘平滑
    pad = smooth_kernel // 2
    kernel = torch.ones(1, 1, smooth_kernel, smooth_kernel, device=device)
    kernel = kernel / kernel.sum()

    att_new = F.conv2d(att_new, kernel, padding=pad)
    att_new = att_new / (att_new.amax(dim=(-2,-1), keepdim=True) + eps)
    att_new = att_new.squeeze(1)

    # 4. CDF 构建
    map_sx = att_new.max(dim=1)[0]
    map_sy = att_new.max(dim=2)[0]

    map_sx = map_sx / (map_sx.sum(dim=1, keepdim=True) + eps)
    map_sy = map_sy / (map_sy.sum(dim=1, keepdim=True) + eps)

    cdf_x = torch.cumsum(map_sx, dim=1)
    cdf_y = torch.cumsum(map_sy, dim=1)

    uniform = torch.linspace(0, 1, out_size, device=device)

    index_x = torch.zeros(B, out_size, dtype=torch.long, device=device)
    index_y = torch.zeros(B, out_size, dtype=torch.long, device=device)

    for i in range(B):
        index_x[i] = torch.searchsorted(cdf_x[i], uniform)
        index_y[i] = torch.searchsorted(cdf_y[i], uniform)

    index_x = index_x.clamp(0, W-1)
    index_y = index_y.clamp(0, H-1)

    # 5. grid_sample
    gx = 2.0 * index_x.float() / (W-1) - 1.0
    gy = 2.0 * index_y.float() / (H-1) - 1.0

    grids = []
    for i in range(B):
        gy_i, gx_i = torch.meshgrid(gy[i], gx[i], indexing='ij')
        grids.append(torch.stack([gx_i, gy_i], dim=-1))

    grid = torch.stack(grids, dim=0)

    sampled_data = F.grid_sample(
        data,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    if return_bbox:
        return sampled_data, att.squeeze(1), att_new, bboxes
    else:
        return sampled_data, att_new

def denorm(img):
    """ImageNet 反归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3,1,1)
    img = img * std + mean
    return img.clamp(0,1)

def visualize_images(
    imgs_s1,
    local_imgs_s1,
    heatmaps,
    heatmaps_new,
    epoch,
    filename,
    save_path,
    bboxes=None,
    alpha=0.5
):
    """
    Args:
        imgs_s1        : [B,3,H,W] 原图
        local_imgs_s1  : [B,3,H,W] 局部图
        heatmaps       : [B,H,W] attention / att_new
        bboxes         : list of (x1,y1,x2,y2) or None
    """

    os.makedirs(save_path, exist_ok=True)
    B = imgs_s1.size(0)

    imgs_s1 = denorm(imgs_s1).detach().cpu()
    local_imgs_s1 = denorm(local_imgs_s1).detach().cpu()
    heatmaps = heatmaps.detach().cpu()
    heatmaps_new = heatmaps_new.detach().cpu()

    for i in range(B):
        fig, axs = plt.subplots(1, 4, figsize=(16,4))

        # 原图
        axs[0].imshow(imgs_s1[i].permute(1,2,0))
        axs[0].set_title("Original")
        axs[0].axis("off")

        # 原图 + Attention+ ROI bbox（可选）
        axs[1].imshow(imgs_s1[i].permute(1,2,0))
        axs[1].imshow(heatmaps[i], cmap="jet", alpha=alpha)
        if bboxes is not None and bboxes[i] is not None:
            x1, y1, x2, y2 = bboxes[i]
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            axs[1].add_patch(rect)

        axs[1].set_title("Attention Overlay")
        axs[1].axis("off")

        # 原图 + Attention_new + ROI bbox（可选）
        axs[2].imshow(imgs_s1[i].permute(1,2,0))
        axs[2].imshow(heatmaps_new[i], cmap="jet", alpha=alpha)

        if bboxes is not None and bboxes[i] is not None:
            x1, y1, x2, y2 = bboxes[i]
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            axs[2].add_patch(rect)

        axs[2].set_title("ROI + Attention")
        axs[2].axis("off")

        # 局部图
        axs[3].imshow(local_imgs_s1[i].permute(1,2,0))
        axs[3].set_title("CDF Sampled")
        axs[3].axis("off")

        save_file = os.path.join(
            save_path,
            f"{filename}_ep{epoch:03d}_idx{i}.png"
        )
        plt.tight_layout()
        plt.savefig(save_file, dpi=150)
        plt.close()