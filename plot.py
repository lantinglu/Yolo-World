import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from all import SingleSplitCOCODataset, YOLOWorldDetector, collate_fn, IMG_DIR, ANN_FILE, INPUT_SIZE, DEVICE

CHECKPOINT_PATH = "yolo_world_last.pth"
CONF_THRESHOLD = 0.05
VIS_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_and_sigmoid(tensor, name="Tensor"):
    min_val, max_val = tensor.min().item(), tensor.max().item()
    if min_val >= 0 and max_val <= 1.0:
        return tensor
    else:
        return torch.sigmoid(tensor)


def post_process_custom(boxes, obj, logits, conf_thres=0.001, iou_thres=0.6):
    batch_preds = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        score_obj = check_and_sigmoid(obj[i], "Obj_Score").view(-1)
        score_cls = check_and_sigmoid(logits[i], "Cls_Score")

        max_cls_score, cls_idx = score_cls.max(dim=1)
        final_score = score_obj * max_cls_score

        mask = final_score > conf_thres
        box = box[mask]
        final_score = final_score[mask]
        cls_idx = cls_idx[mask]

        if len(box) == 0:
            batch_preds.append({'boxes': [], 'scores': [], 'labels': []})
            continue

        # NMS
        keep = torch.ops.torchvision.nms(box, final_score, iou_thres)
        batch_preds.append({
            'boxes': box[keep],
            'scores': final_score[keep],
            'labels': cls_idx[keep]
        })
    return batch_preds


def draw_boxes_on_ax(ax, boxes, color, scores=None, label_text=""):
    if boxes is None or len(boxes) == 0:
        return 0

    count = 0
    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu() if scores is not None else None

    for i in range(len(boxes_cpu)):
        if scores_cpu is not None and scores_cpu[i] < VIS_THRESHOLD:
            continue

        count += 1
        x1, y1, x2, y2 = boxes_cpu[i]
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        display_text = label_text
        if scores_cpu is not None:
            display_text = f"{scores_cpu[i]:.2f}"

        if display_text:
            ax.text(x1, y1, display_text, color='white' if color == 'red' else 'black',
                    fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.5, pad=0, edgecolor='none'))
    return count


def visualize_three_views(img_tensor, pred, target, base_idx):
    print(f"\n--- Generating 3 Views for Image Index {base_idx} ---")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = img_tensor.cpu() * std + mean
    img_vis = img_vis.permute(1, 2, 0).numpy()
    img_vis = np.clip(img_vis, 0, 1)

    base_filename = os.path.join(OUTPUT_DIR, f"debug_idx_{base_idx}")

    plt.figure(figsize=(12, 12))
    plt.imshow(img_vis)
    ax = plt.gca()
    count = draw_boxes_on_ax(ax, pred['boxes'], color='red', scores=pred['scores'])
    plt.axis('off')
    plt.title(f"View 1: Predictions Only (Red, Thres > {VIS_THRESHOLD}), Count: {count}")
    save_path = f"{base_filename}_1_pred_red.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved View 1: {save_path}")

    plt.figure(figsize=(12, 12))
    plt.imshow(img_vis)
    ax = plt.gca()
    count = draw_boxes_on_ax(ax, target['boxes'], color='#00FF00', label_text="GT")
    plt.axis('off')
    plt.title(f"View 2: Ground Truth Only (Green), Count: {count}")
    save_path = f"{base_filename}_2_gt_green.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved View 2: {save_path}")

    plt.figure(figsize=(12, 12))
    plt.imshow(img_vis)
    ax = plt.gca()
    draw_boxes_on_ax(ax, target['boxes'], color='#00FF00', label_text="GT")
    draw_boxes_on_ax(ax, pred['boxes'], color='red', scores=pred['scores'])
    plt.axis('off')
    plt.title(f"View 3: Overlay Comparison (Green=GT, Red=Pred)")
    save_path = f"{base_filename}_3_overlay.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved View 3: {save_path}")
    print("--- Done ---")


def main():
    print("Loading Dataset...")
    full_dataset = SingleSplitCOCODataset(IMG_DIR, ANN_FILE, transform_size=INPUT_SIZE)

    idx = random.randint(0, len(full_dataset) - 1)
    print(f"Selected Image Index: {idx}")

    img, texts, gt = full_dataset[idx]
    imgs, texts_batch, gts = collate_fn([(img, texts, gt)])
    imgs = imgs.to(DEVICE)

    print("Loading Model...")
    cfg_vis = {"model_name": "resnet50"}
    cfg_txt = {"model_name": "openai/clip-vit-base-patch32", "freeze_all": True}

    model = YOLOWorldDetector(
        config_vision=cfg_vis, config_text=cfg_txt,
        neck_in_channels=[512, 1024, 2048],
        neck_out_channels=[256, 512, 1024]
    ).to(DEVICE)

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Running Inference...")
    with torch.no_grad():
        boxes_pred, obj_pred, _, logits_pred = model(imgs, [texts_batch[0]])

        preds = post_process_custom(boxes_pred, obj_pred, logits_pred,
                                    conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)

    target_dict = {"boxes": gts[0]["boxes"], "labels": gts[0]["labels"]}
    visualize_three_views(imgs[0], preds[0], target_dict, base_idx=idx)
    print(f"\nResult images saved to directory: {OUTPUT_DIR}/")


if __name__ == "__main__":

    main()