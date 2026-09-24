import torch

from clip_text_encoder import TextEncoder
from my_yolo_world_backbone import MyVisionBackbone
from repvl_pan import RepVLPANDual, ConvBNAct

class DummyBasePAN(torch.nn.Module):
    def __init__(self, chans=[768, 768, 768]):
        super().__init__()
        self.conv3 = ConvBNAct(chans[0], chans[0])
        self.conv4 = ConvBNAct(chans[1], chans[1])
        self.conv5 = ConvBNAct(chans[2], chans[2])

    def forward(self, c3, c4, c5):
        return self.conv3(c3), self.conv4(c4), self.conv5(c5)

image_backbone_config = {
    'model_name': 'openai/clip-vit-base-patch16',
    'out_indices': (3, 6, 9, 12),
    'norm_eval': True,
    'frozen_prefixes': ('vision_model.embeddings',),
}

image_backbone = MyVisionBackbone(**image_backbone_config)
text_encoder = TextEncoder()

c_chans = [768, 768, 768]
p_chans = [768, 768, 768]
d_text = 512

base_pan = DummyBasePAN(c_chans)
repvl_pan = RepVLPANDual(base_pan, c_chans, p_chans, d_text)

batch_size = 2
image = torch.randn(batch_size, 3, 224, 224)
texts = [["texttexttexttexttexttexttexttexttext", "asdfgasdfasfasdasdfasdf"]] * batch_size

img_feats = image_backbone(image)
txt_feats = text_encoder(texts)

C3, C4, C5 = img_feats[-3:]
T = txt_feats.mean(dim=0)  # [N, D]

print("Image features shapes:", [f.shape for f in (C3, C4, C5)])
print("Text features shape:", txt_feats.shape)

assert [C3.shape[1], C4.shape[1], C5.shape[1]] == c_chans
assert txt_feats.shape[-1] == d_text

with torch.no_grad():
    pan_outputs, updated_T = repvl_pan(C3, C4, C5, T)
print("RepVL-PAN forward success!")
print("PAN outputs shapes:", [o.shape for o in pan_outputs])
print("Updated text shape:", updated_T.shape)

repvl_pan.fold_for_vocab(T)
folded_outputs = repvl_pan.forward_folded(C3, C4, C5)
print("Folded outputs shapes:", [o.shape for o in folded_outputs])

print("Validation complete: TextEncoder outputs match RepVL-PAN inputs!")