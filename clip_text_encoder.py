import itertools
from typing import List

from torch import nn
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch16',
        freeze_all: bool = True,
        return_mask: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = CLIPTextConfig.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name, config=config)
        self.return_mask = return_mask

        if freeze_all:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, texts: List[List[str]]):

        num_per_sample = [len(t) for t in texts]
        if len(set(num_per_sample)) != 1:
            raise ValueError("All samples must have the same number of prompts.")
        B, N = len(texts), num_per_sample[0]

        flat_texts = list(itertools.chain.from_iterable(texts))
        inputs = self.tokenizer(
            flat_texts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(**inputs)
        embeds = outputs.text_embeds  # [B*N, D]
        embeds = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-6)
        embeds = embeds.view(B, N, -1)  # [B, N, D]

        if not self.return_mask:
            return embeds

        attn_mask = inputs["attention_mask"]
        valid_mask = attn_mask.any(dim=-1).view(B, N)
        return embeds, valid_mask