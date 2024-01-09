import torch
from vit_pytorch.vit_3d import ViT

v = ViT(
    image_size = 240,          # image size
    frames = 154,               # number of frames
    image_patch_size = 120,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 3,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels= 1
)

video = torch.randn(4, 1, 154, 240, 240) # (batch, channels, frames, height, width)

preds = v(video) # (4, 3)

print(preds)


import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)