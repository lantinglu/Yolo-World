import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(c1, c2, k=3, s=1, act=True):
    p = k // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True) if act else nn.Identity(),
    )


def _flatten_hw(xs: List[torch.Tensor]) -> torch.Tensor:
    outs = []
    for x in xs:
        B, C, H, W = x.shape
        outs.append(x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C))
    return torch.cat(outs, dim=1)


def _build_grid(h: int, w: int, stride: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    grid = torch.stack([xs, ys], dim=-1).view(-1, 2).float()
    grid = (grid + 0.5) * stride
    strides = torch.full((grid.shape[0], 1), stride, device=device)
    return torch.cat([grid, strides], dim=-1)


class YOLOWorldHead(nn.Module):
    def __init__(self, chans: List[int], d_text: int = 512, strides: List[int] = (8, 16, 32),
                 num_box_conv: int = 2, num_obj_conv: int = 2, num_emb_conv: int = 2, init_tau: float = 16.0):
        super().__init__()
        self.chans = chans
        self.strides = list(strides)
        self.d = d_text
        self.box_branch = nn.ModuleList()
        self.obj_branch = nn.ModuleList()
        self.emb_branch = nn.ModuleList()
        self.box_out = nn.ModuleList()
        self.obj_out = nn.ModuleList()
        self.emb_out = nn.ModuleList()

        for c in chans:
            self.box_branch.append(nn.Sequential(*[conv_bn_act(c, c) for _ in range(num_box_conv)]))
            self.obj_branch.append(nn.Sequential(*[conv_bn_act(c, c) for _ in range(num_obj_conv)]))
            self.emb_branch.append(nn.Sequential(*[conv_bn_act(c, c) for _ in range(num_emb_conv)]))
            self.box_out.append(nn.Conv2d(c, 4, 1))
            self.obj_out.append(nn.Conv2d(c, 1, 1))
            self.emb_out.append(nn.Conv2d(c, self.d, 1))

        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        for m in self.box_out: nn.init.constant_(m.bias, 1.0)

    def forward(self, feats, T_used, image_size=None):
        device = feats[0].device
        box_maps, obj_maps, emb_maps, grids = [], [], [], []
        for i, f in enumerate(feats):
            _, _, H, W = f.shape
            box_maps.append(self.box_out[i](self.box_branch[i](f)))
            obj_maps.append(self.obj_out[i](self.obj_branch[i](f)))
            emb_maps.append(self.emb_out[i](self.emb_branch[i](f)))
            grids.append(_build_grid(H, W, self.strides[i], device))

        box_raw = _flatten_hw(box_maps)
        obj_log = _flatten_hw(obj_maps)
        f_obj = F.normalize(_flatten_hw(emb_maps), dim=-1)
        Tn = F.normalize(T_used, dim=-1)
        logits = self.tau * torch.einsum("bnd,kd->bnk", f_obj, Tn)
        obj = torch.sigmoid(obj_log)
        boxes = self.decode_boxes_xywh(box_raw, grids, image_size) if image_size else box_raw
        return boxes, obj, f_obj, logits

    def decode_boxes_xywh(self, box_raw, grids, image_size):
        H_img, W_img = image_size
        device = box_raw.device
        grid_all = torch.cat(grids, dim=0).to(device)

        anchor_points = grid_all[:, :2]
        strides = grid_all[:, 2:3]

        tx, ty, tw, th = torch.chunk(box_raw, 4, dim=-1)

        cx = anchor_points[:, 0].view(1, -1, 1) + (torch.sigmoid(tx) - 0.5) * strides.view(1, -1, 1)
        cy = anchor_points[:, 1].view(1, -1, 1) + (torch.sigmoid(ty) - 0.5) * strides.view(1, -1, 1)
        w = torch.exp(tw) * strides.view(1, -1, 1)
        h = torch.exp(th) * strides.view(1, -1, 1)

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes = torch.cat([x1, y1, x2, y2], dim=-1)

        final_boxes = boxes.clone()
        final_boxes[..., 0] = boxes[..., 0].clamp(0, W_img - 1)
        final_boxes[..., 1] = boxes[..., 1].clamp(0, H_img - 1)
        final_boxes[..., 2] = boxes[..., 2].clamp(0, W_img - 1)
        final_boxes[..., 3] = boxes[..., 3].clamp(0, H_img - 1)

        return final_boxes