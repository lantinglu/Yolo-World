import torch
from torch.utils.data import DataLoader, Subset
from torchvision import ops
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from all import SingleSplitCOCODataset, YOLOWorldDetector, collate_fn, IMG_DIR, ANN_FILE, INPUT_SIZE, DEVICE

CHECKPOINT_PATH = "yolo_world_last.pth"
YOLO_V8_MODEL = "yolov8s.pt"
NUM_TEST_IMAGES = 500
CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.6


def post_process_custom(boxes, obj, logits, conf_thres=0.001, iou_thres=0.6):

    batch_preds = []

    for i in range(boxes.shape[0]):
        box = boxes[i]

        score_obj = torch.sigmoid(obj[i]).view(-1)
        score_cls = torch.sigmoid(logits[i])

        max_cls_score, cls_idx = score_cls.max(dim=1)

        final_score = score_obj * max_cls_score

        mask = final_score > conf_thres
        box = box[mask]
        final_score = final_score[mask]
        cls_idx = cls_idx[mask]

        if len(box) == 0:
            batch_preds.append({
                'boxes': torch.tensor([], device=boxes.device),
                'scores': torch.tensor([], device=boxes.device),
                'labels': torch.tensor([], device=boxes.device)
            })
            continue

        keep = ops.nms(box, final_score, iou_thres)

        batch_preds.append({
            'boxes': box[keep],
            'scores': final_score[keep],
            'labels': cls_idx[keep]
        })

    return batch_preds


def main():
    print(f"Starting comparison on {DEVICE}...")

    full_dataset = SingleSplitCOCODataset(IMG_DIR, ANN_FILE, transform_size=INPUT_SIZE)
    indices = list(range(min(NUM_TEST_IMAGES, len(full_dataset))))
    subset_dataset = Subset(full_dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    print("Loading Custom YOLO-World Model...")
    cfg_vis = {"model_name": "resnet50"}
    cfg_txt = {"model_name": "openai/clip-vit-base-patch32", "freeze_all": True}

    custom_model = YOLOWorldDetector(
        config_vision=cfg_vis, config_text=cfg_txt,
        neck_in_channels=[512, 1024, 2048],
        neck_out_channels=[256, 512, 1024]
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    custom_model.load_state_dict(checkpoint)
    custom_model.eval()

    print(f"Loading Ultralytics {YOLO_V8_MODEL}...")
    yolo8_model = YOLO(YOLO_V8_MODEL)

    metric_custom = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    metric_yolo8 = MeanAveragePrecision(iou_type="bbox", class_metrics=False)

    print(f"Running Inference on {len(indices)} images...")

    with torch.no_grad():
        for imgs, texts, gts in tqdm(loader):
            imgs = imgs.to(DEVICE)

            boxes_pred, obj_pred, _, logits_pred = custom_model(imgs, [texts[0]])

            preds_custom = post_process_custom(boxes_pred, obj_pred, logits_pred,
                                               conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)

            target_list = []
            for t in gts:
                target_list.append({
                    "boxes": t["boxes"].to(DEVICE),
                    "labels": t["labels"].to(DEVICE)
                })

            metric_custom.update(preds_custom, target_list)

            mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
            imgs_unnorm = (imgs * std + mean) * 255.0
            imgs_unnorm = imgs_unnorm.clamp(0, 255).byte()

            yolo_results = yolo8_model(imgs_unnorm, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

            preds_yolo = []
            for res in yolo_results:
                preds_yolo.append({
                    'boxes': res.boxes.xyxy.to(DEVICE),
                    'scores': res.boxes.conf.to(DEVICE),
                    'labels': res.boxes.cls.long().to(DEVICE)
                })

            metric_yolo8.update(preds_yolo, target_list)

    print("Computing mAP...")
    result_custom = metric_custom.compute()
    result_yolo8 = metric_yolo8.compute()

    map50_custom = result_custom['map_50'].item()
    map50_yolo8 = result_yolo8['map_50'].item()

    print(f"\nResults (on {NUM_TEST_IMAGES} images):")
    print(f"Custom Model mAP@50: {map50_custom:.4f}")
    print(f"YOLOv8s    mAP@50: {map50_yolo8:.4f}")

    models = ['Custom YOLO-World', f'YOLOv8s']
    maps = [map50_custom, map50_yolo8]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, maps, color=['#1f77b4', '#ff7f0e'])

    plt.ylabel('mAP@50')
    plt.title(f'Detection Performance Comparison\n(Subset: {NUM_TEST_IMAGES} images)')
    plt.ylim(0, 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

    plt.savefig('comparison_result.png')
    print("Chart saved as 'comparison_result.png'")
    plt.show()


if __name__ == "__main__":
    main()