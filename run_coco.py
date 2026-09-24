import os
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from transformers import CLIPProcessor

from clip_text_encoder import TextEncoder
from my_yolo_world_backbone import MyVisionBackbone

# 数据集位置
COCO_ROOT = "./data/coco"
# 数据集中的图片的文件夹的位置
IMG_DIR = os.path.join(COCO_ROOT, "val2017")
# 标签的位置
ANN_FILE = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")

# 输出位置
OUTPUT_DIR = "./coco_embeddings_output"

MODEL_NAME = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using Device: {DEVICE}")

    print(f"Loading COCO annotations from {ANN_FILE}...")
    coco = COCO(ANN_FILE)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    class_names = [c['name'] for c in cats]

    prompts = [[name] for name in class_names]

    print(f"Found {len(class_names)} categories (e.g., {class_names[:3]}).")

    print("Loading Text Encoder...")
    text_encoder = TextEncoder(model_name=MODEL_NAME).to(DEVICE)
    text_encoder.eval()

    print("Generating text embeddings...")
    with torch.no_grad():
        text_feats = text_encoder(prompts)
        text_feats = text_feats.squeeze(1)

    text_save_path = os.path.join(OUTPUT_DIR, "coco_text_embeddings.pt")
    torch.save({
        "cat_ids": cat_ids,
        "class_names": class_names,
        "embeddings": text_feats.cpu()
    }, text_save_path)
    print(f"Text embeddings saved to: {text_save_path}")

    del text_encoder
    torch.cuda.empty_cache()
    print("Loading Vision Backbone...")
    vision_model = MyVisionBackbone(
        model_name=MODEL_NAME,
        out_indices=(0, 1, 2, 3),
        norm_eval=True
    ).to(DEVICE)
    vision_model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    img_ids = coco.getImgIds()
    images_info = coco.loadImgs(img_ids)

    print(f"Starting extraction for {len(images_info)} images...")

    image_results = {}


    with torch.no_grad():
        for img_info in tqdm(images_info):
            file_name = img_info['file_name']
            img_path = os.path.join(IMG_DIR, file_name)

            if not os.path.exists(img_path):
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")

                inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
                pixel_values = inputs.pixel_values

                feats = vision_model(pixel_values)
                feats_cpu = tuple(f.cpu() for f in feats)

                image_results[img_info['id']] = {
                    "file_name": file_name,
                    "features": feats_cpu,
                    "height": img_info['height'],
                    "width": img_info['width']
                }

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    img_save_path = os.path.join(OUTPUT_DIR, "coco_val2017_features.pt")
    torch.save(image_results, img_save_path)

    print(f"Image features saved to: {img_save_path}")
    print("Done.")


if __name__ == "__main__":
    main()