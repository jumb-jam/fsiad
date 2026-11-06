"""DINOv2 feature extraction (rotation and masking live in preprocess.py).

Everything here copies the AnomalyDINO code (src/backbones.py, src/utils.py)
so that the reproduced baseline matches the paper (spec Section 2.5).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .preprocess import background_mask, rotate_image  # noqa: F401  (re-exported)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Extractor:
    """Last-layer, LayerNorm-ed DINOv2 patch tokens (same as `x_norm_patchtokens`)."""

    def __init__(self, model_name: str = "dinov2_vits14", resolution: int = 448,
                 device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).eval().to(device)
        self.patch_size = self.model.patch_size
        self.transform = transforms.Compose([
            transforms.Resize(size=resolution, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def prepare(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize smaller edge, normalise, trim bottom/right to a multiple of the patch size."""
        t = self.transform(Image.fromarray(image_rgb))
        h, w = t.shape[1:]
        h, w = h - h % self.patch_size, w - w % self.patch_size
        return t[:, :h, :w], (h // self.patch_size, w // self.patch_size)

    @torch.inference_mode()
    def extract(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Batch of prepared tensors (all the same size) -> (B, P, D) float32 on CPU."""
        batch = torch.stack(tensors).to(self.device)
        tokens = self.model.get_intermediate_layers(batch)[0]   # n=1, norm=True
        return tokens.float().cpu()
