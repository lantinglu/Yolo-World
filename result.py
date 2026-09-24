import torch

ckpt = torch.load(r"D:\UW\大四\25AU\CSE455\final project\result\yolo_world_last.pth", 
                  map_location="cpu")

print(ckpt.keys())
