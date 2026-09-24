import torch
import torch.nn as nn
from typing import List

# 导入你的模块
from my_yolo_world_backbone import MyVisionBackbone, MyCLIPTextBackbone
from repvl_pan import YOLOWorldPAFPN, RepVLPANDual
from heads import YOLOWorldHead

class YOLOWorldDetector(nn.Module):
    def __init__(self,
                 config_vision,
                 config_text,
                 neck_in_channels=[512, 1024, 2048],
                 neck_out_channels=[256, 512, 1024],
                 num_classes=80):
        super().__init__()

        self.vision_backbone = MyVisionBackbone(**config_vision)
        self.text_backbone = MyCLIPTextBackbone(**config_text)

        base_pan = YOLOWorldPAFPN(
            in_channels=neck_in_channels,
            out_channels=neck_out_channels,
            num_csp_blocks=3
        )

        self.neck = RepVLPANDual(
            base_pan=base_pan,
            c_chans=neck_in_channels,
            p_chans=neck_out_channels,
            d_text=512
        )

        self.head = YOLOWorldHead(
            chans=neck_out_channels,
            d_text=512,
            strides=[8, 16, 32]
        )

    def forward(self, images, texts: List[List[str]]):
        img_feats = self.vision_backbone(images)

        txt_feats = self.text_backbone(texts)
        t_batch = txt_feats[0]

        (p3, p4, p5), t_updated = self.neck(img_feats[0], img_feats[1], img_feats[2], t_batch)

        H, W = images.shape[2], images.shape[3]
        return self.head([p3, p4, p5], t_updated, image_size=(H, W))