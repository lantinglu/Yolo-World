import torch.nn as nn
import torch.nn.functional as F
from box_utils import bbox_iou

class DetectionLoss(nn.Module):
    def __init__(self, box_weight=5.0, obj_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

    def box_loss(self, pred_box, tgt_box):
        pred_box = pred_box.reshape(-1, 4)
        tgt_box = tgt_box.reshape(-1, 4)

        iou = bbox_iou(pred_box, tgt_box, xywh=False, CIoU=True)

        loss = 1.0 - iou

        return loss

    def forward(self, preds, assigned):
        pred_box = preds["box"]
        pred_obj = preds["obj"]
        pred_cls = preds["cls"]

        tgt_box = assigned["boxes"]
        tgt_obj = assigned["obj"]
        tgt_cls = assigned["cls"]

        l_box = self.box_loss(pred_box, tgt_box).mean()
        l_obj = self.bce(pred_obj, tgt_obj).mean()
        l_cls = self.bce(pred_cls, tgt_cls).mean()

        total = (
            self.box_weight * l_box +
            self.obj_weight * l_obj +
            self.cls_weight * l_cls
        )

        return {
            "loss_total": total,
            "loss_box": l_box.detach(),
            "loss_obj": l_obj.detach(),
            "loss_cls": l_cls.detach()
        }



class TextRegionContrastLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, region_feat, text_feat):
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)
        t = F.normalize(text_feat, dim=-1)
        t = t.expand(region_feat.size(0), -1)

        r = F.normalize(region_feat, dim=-1)

        sim = (r * t).sum(dim=-1) / self.temperature
        loss = - sim.mean()
        return loss

