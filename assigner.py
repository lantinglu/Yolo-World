import torch
import torch.nn.functional as F
from typing import List, Tuple

@torch.no_grad()
def _build_centers(feat_shapes: List[Tuple[int,int]], strides: Tuple[int,int,int], device):
    centers=[]; s_all=[]
    for (H,W), s in zip(feat_shapes, strides):
        ys, xs = torch.meshgrid(torch.arange(H,device=device),
                                torch.arange(W,device=device), indexing='ij')
        g = torch.stack([xs, ys], -1).reshape(-1,2).float()
        g = (g + 0.5) * float(s)   # 像素中心
        centers.append(g); s_all.append(torch.full((H*W,), float(s), device=device))
    return torch.cat(centers,0), torch.cat(s_all,0)

@torch.no_grad()
def _iou_xyxy(a, b):
    tl = torch.max(a[:,None,:2], b[None,:, :2])
    br = torch.min(a[:,None,2:], b[None,:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[...,0]*wh[...,1]
    area_a = (a[:,2]-a[:,0]).clamp(0) * (a[:,3]-a[:,1]).clamp(0)
    area_b = (b[:,2]-b[:,0]).clamp(0) * (b[:,3]-b[:,1]).clamp(0)
    union = area_a[:,None] + area_b[None,:] - inter + 1e-7
    return inter/union  # (N,M)

@torch.no_grad()
def task_aligned_assigner_lite(
    pred_boxes: torch.Tensor,
    logits: torch.Tensor,
    feat_shapes: List[Tuple[int,int]],  # [(H3,W3),(H4,W4),(H5,W5)]
    strides: Tuple[int,int,int] = (8,16,32),
    gts: List[dict] = None,
    topk: int = 13,
    alpha: float = 1.0,
    beta: float = 6.0,
    center_radius: float = 2.5,
    return_soft: bool = True
):
    """
    返回：
      pos_mask:   (B,N)   bool
      matched_ix: (B,N)   long  正样本匹配的 GT 索引，负样本 -1
      obj_t:      (B,N,1) 0/1
      cls_t:      (B,N)   long  正样本处为 [0..K-1]
      assigned_scores (可选): (B,N,K) 软标签，负样本为 0
    """
    device = pred_boxes.device
    B,N,_ = pred_boxes.shape
    _,_,K = logits.shape

    centers, s_all = _build_centers(feat_shapes, strides, device)
    pred_scores = F.softmax(logits, dim=-1)

    pos_mask   = torch.zeros(B,N, dtype=torch.bool, device=device)
    matched_ix = torch.full((B,N), -1, dtype=torch.long, device=device)
    obj_t      = torch.zeros(B,N,1, device=device)
    cls_t      = torch.zeros(B,N,   dtype=torch.long, device=device)
    assigned_scores = torch.zeros(B,N,K, device=device) if return_soft else None

    for b in range(B):
        gt_boxes = gts[b]["boxes"].to(device)
        gt_cls   = gts[b]["labels"].to(device)
        M = gt_boxes.size(0)
        if M == 0:
            continue

        cx = (gt_boxes[:,0] + gt_boxes[:,2]) * 0.5
        cy = (gt_boxes[:,1] + gt_boxes[:,3]) * 0.5
        r  = center_radius * s_all
        in_center = (centers[:,0][:,None] > (cx[None,:]-r[:,None])) & \
                    (centers[:,0][:,None] < (cx[None,:]+r[:,None])) & \
                    (centers[:,1][:,None] > (cy[None,:]-r[:,None])) & \
                    (centers[:,1][:,None] < (cy[None,:]+r[:,None]))
        in_gt = (centers[:,0][:,None] > gt_boxes[None,:,0]) & \
                (centers[:,0][:,None] < gt_boxes[None,:,2]) & \
                (centers[:,1][:,None] > gt_boxes[None,:,1]) & \
                (centers[:,1][:,None] < gt_boxes[None,:,3])
        prior_mask = in_center & in_gt

        iou_nm = _iou_xyxy(pred_boxes[b], gt_boxes)
        cls_scores_nm = pred_scores[b].gather(1, gt_cls.view(1,-1).expand(N,M))
        align_nm = (cls_scores_nm.clamp(min=1e-6) ** alpha) * (iou_nm.clamp(min=1e-6) ** beta)

        cand = torch.zeros_like(prior_mask)
        for j in range(M):
            if prior_mask[:,j].any():
                scores_j = align_nm[:,j].clone()
                scores_j[~prior_mask[:,j]] = -1.0  # 排除不在先验内的 prior
                k = min(topk, scores_j.numel())
                top_idx = torch.topk(scores_j, k=k, dim=0).indices
                keep = torch.zeros_like(scores_j, dtype=torch.bool)
                keep[top_idx] = True
                cand[:,j] = keep & prior_mask[:,j]

        any_pos = cand.any(dim=1)
        pos_mask[b, any_pos] = True
        obj_t[b, any_pos, 0] = 1.0

        if any_pos.any():
            sel_align = align_nm[any_pos]
            sel_align = torch.where(cand[any_pos], sel_align, torch.full_like(sel_align, -1.0))
            gt_choice = sel_align.argmax(dim=1)
            matched_ix[b, any_pos] = gt_choice
            cls_t[b, any_pos] = gt_cls[gt_choice]

            if return_soft:
                P = any_pos.sum().item()
                rows = torch.arange(P, device=device)
                soft = torch.zeros(P, K, device=device)
                chosen_align = sel_align[rows, gt_choice].clamp(min=1e-6)
                chosen_iou   = iou_nm[any_pos][rows, gt_choice].clamp(min=1e-6)
                score = (chosen_align * chosen_iou)
                score = score / (score.max() + 1e-6)
                soft[rows, cls_t[b, any_pos]] = score
                assigned_scores[b, any_pos] = soft

    if return_soft:
        return pos_mask, matched_ix, obj_t, cls_t, assigned_scores
    else:
        return pos_mask, matched_ix, obj_t, cls_t
