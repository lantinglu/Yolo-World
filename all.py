import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

from detector import YOLOWorldDetector
from assigner import task_aligned_assigner_lite
from loss import DetectionLoss

COCO_ROOT = "./data/coco"
IMG_DIR = os.path.join(COCO_ROOT, "val2017")
ANN_FILE = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")

# 这里，如果显存不够高，调这里就行
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = (640, 640)

imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)



class SingleSplitCOCODataset(Dataset):
    def __init__(self, root, ann_file, transform_size=(640, 640)):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform_size = transform_size
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.vocab = [c['name'] for c in cats]
        self.cat_map = {cid: i for i, cid in enumerate(sorted(self.coco.getCatIds()))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, path)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return torch.randn(3, *self.transform_size), self.vocab, {"boxes": torch.zeros(0, 4),
                                                                      "labels": torch.zeros(0, dtype=torch.long)}

        w_orig, h_orig = image.size
        image = image.resize(self.transform_size)

        img_tensor = torch.from_numpy(
            torch.tensor(list(image.getdata()), dtype=torch.float32).view(self.transform_size[1],
                                                                          self.transform_size[0], 3).numpy()
        ).permute(2, 0, 1) / 255.0
        img_tensor = imagenet_normalize(img_tensor)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        boxes, labels = [], []
        sx, sy = self.transform_size[0] / w_orig, self.transform_size[1] / h_orig

        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
            x1, y1 = max(0, min(x1, 640)), max(0, min(y1, 640))
            x2, y2 = max(0, min(x2, 640)), max(0, min(y2, 640))
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_map[ann['category_id']])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return img_tensor, self.vocab, target


def collate_fn(batch):
    imgs, vocabs, targets = zip(*batch)
    return torch.stack(imgs), list(vocabs), list(targets)


def main():
    print(f"Using device: {DEVICE}")
    full_dataset = SingleSplitCOCODataset(IMG_DIR, ANN_FILE, transform_size=INPUT_SIZE)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, drop_last=True)

    cfg_vis = {"model_name": "resnet50"}
    cfg_txt = {"model_name": "openai/clip-vit-base-patch32", "freeze_all": True}

    model = YOLOWorldDetector(
        config_vision=cfg_vis, config_text=cfg_txt,
        neck_in_channels=[512, 1024, 2048],
        neck_out_channels=[256, 512, 1024]
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = DetectionLoss().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')


    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
        epoch_loss = 0.0

        for i, (imgs, texts, gts) in pbar:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                boxes_pred, obj_pred, _, logits_pred = model(imgs, [texts[0]])
                feat_shapes = [(imgs.shape[2] // s, imgs.shape[3] // s) for s in model.head.strides]

                with torch.no_grad():
                    pos_mask, matched_ix, obj_t, cls_t = task_aligned_assigner_lite(
                        pred_boxes=boxes_pred.detach(),
                        logits=logits_pred.detach().float(),
                        feat_shapes=feat_shapes, strides=model.head.strides, gts=gts, return_soft=False
                    )

                # Loss 计算
                mask = pos_mask.view(-1)
                preds_pack = {
                    "box": boxes_pred.reshape(-1, 4)[mask],
                    "obj": obj_pred.reshape(-1, 1),
                    "cls": logits_pred.reshape(-1, logits_pred.shape[-1])[mask]
                }

                batch_tgt_boxes, batch_tgt_cls = [], []
                for b in range(BATCH_SIZE):
                    if len(gts) <= b: break
                    b_mask = pos_mask[b]
                    b_idx = matched_ix[b][b_mask]
                    if len(b_idx) > 0:
                        batch_tgt_boxes.append(gts[b]["boxes"].to(DEVICE)[b_idx])
                        t_cls = torch.zeros(len(b_idx), logits_pred.shape[-1], device=DEVICE)
                        t_cls[range(len(b_idx)), gts[b]["labels"].to(DEVICE)[b_idx]] = 1.0
                        batch_tgt_cls.append(t_cls)

                if len(batch_tgt_boxes) == 0:
                    loss_val = criterion.bce(obj_pred.view(-1, 1), obj_t.view(-1, 1)).mean()
                else:
                    target_pack = {
                        "boxes": torch.cat(batch_tgt_boxes),
                        "obj": obj_t.reshape(-1, 1),
                        "cls": torch.cat(batch_tgt_cls)
                    }
                    loss_val = criterion(preds_pack, target_pack)["loss_total"]

            scaler.scale(loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_val.item()
            pbar.set_postfix({"loss": f"{loss_val.item():.4f}", "avg": f"{epoch_loss / (i + 1):.4f}"})

        scheduler.step()
        torch.save(model.state_dict(), "yolo_world_last.pth")


if __name__ == "__main__":
    main()