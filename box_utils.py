import torch
from torchvision.ops import nms
import math


def yolo_world_postprocess(boxes, obj, cls_scores, iou_thr=0.5, score_thr=0.25):
    boxes = boxes[0]
    obj = obj[0].squeeze(-1)
    cls_scores = cls_scores[0]

    cls_prob, cls_ids = cls_scores.max(dim=-1)
    final_scores = obj * cls_prob

    keep = final_scores > score_thr
    boxes = boxes[keep]
    final_scores = final_scores[keep]
    cls_ids = cls_ids[keep]

    keep_idx = nms(boxes, final_scores, iou_thr)
    return boxes[keep_idx], final_scores[keep_idx], cls_ids[keep_idx]


def draw_boxes(img, boxes, scores, cls_ids, vocab):
    import cv2
    img_draw = img.copy()
    for box, sc, cid in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = vocab[cid] if cid < len(vocab) else str(cid)
        text = f"{label} {sc:.2f}"
        cv2.putText(img_draw, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_draw


def bbox_iou(box1, box2, xywh=False, CIoU=True, eps=1e-7):
    """
    计算 CIoU
    box1: (N, 4)
    box2: (N, 4)
    """
    if xywh:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared

        rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 +
                (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


