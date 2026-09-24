
import torch.nn as nn
import torchvision
import itertools
from typing import List, Optional, Sequence
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection


class MyVisionBackbone(nn.Module):
    def __init__(self, model_name="resnet50", **kwargs):
        super().__init__()
        print(f"Loading ResNet50 weights...")
        resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for p in self.stem.parameters(): p.requires_grad = False
        for p in self.layer1.parameters(): p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c3, c4, c5]


class MyCLIPTextBackbone(nn.Module):
    def __init__(
            self,
            model_name: str,
            freeze_all: bool = True,
            return_mask: bool = False,
            init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cfg = CLIPTextConfig.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name, config=cfg)
        self.return_mask = return_mask

        if freeze_all:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, texts: List[List[str]]):
        num_per_sample = [len(t) for t in texts]
        assert len(set(num_per_sample)) == 1, "Each sample must have same #prompts."
        N = num_per_sample[0]
        flat = list(itertools.chain.from_iterable(texts))

        encoded = self.tokenizer(
            flat, padding=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(**encoded)
        feats = outputs.text_embeds
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
        feats = feats.view(-1, N, feats.size(-1))

        if not self.return_mask:
            return feats

        attn_mask = encoded["attention_mask"]
        valid_mask = attn_mask.any(dim=-1).view(-1, N)
        return feats, valid_mask