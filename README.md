# YOLO-World with Multi-Scale Text Adaptation

This project implements a simplified YOLO-World pipeline for open-vocabulary object detection. Given an image and a list of category names such as `person`, `bicycle`, and `dog`, the model predicts bounding boxes and scores for objects that match those names.

## Approach

I wanted to explore how text prompts could guide object detection across multiple image resolutions. I used CLIP to encode category names and added a TextAdapter at three levels of the CNN backbone (C3, C4, and C5). This lets the text embeddings guide both the higher-resolution features that preserve spatial detail and the coarser features that capture more semantic information.

- **Image encoder** extracts feature maps at three spatial resolutions.
- **Text encoder** converts category names into L2-normalized CLIP embeddings.
- **TextAdapter and RepVL-PAN** use the text embeddings to modulate visual features and fuse information across scales.
- **Contrastive detection head** scores candidate detections against the prompted category embeddings.

My main contribution was integrating the TextAdapter at all three feature levels and building the training and evaluation workflow for the resulting detector.

## Evaluation

I evaluated the detector on a subset of COCO and used YOLOv8s as the baseline. The metric was **mAP@50**, which averages per-class average precision using a bounding-box intersection-over-union threshold of 0.50.

| Model | COCO subset mAP@50 |
| --- | ---: |
| YOLOv8s baseline | 54.53% |
| YOLO-World with multi-scale TextAdapter | **61.28%** |

The full pipeline scored **6.75 percentage points** higher than the YOLOv8s baseline on this subset. An ablation using the same architecture with the TextAdapter removed would isolate the adapter’s contribution.
