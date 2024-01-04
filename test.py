import torch
from vit_pytorch.vit_3d import ViT

v = ViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)

print(preds)