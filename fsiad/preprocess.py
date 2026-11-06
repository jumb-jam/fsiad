"""Torch-free preprocessing copied from AnomalyDINO: rotation and PCA background mask."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.decomposition import PCA


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, same size, corners filled by reflection (AnomalyDINO utils.py)."""
    centre = tuple(np.array(image.shape[1::-1]) / 2)
    rot = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(image, rot, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_DEFAULT)


def background_mask(features: np.ndarray, grid: tuple[int, int], threshold: float = 10.0,
                    kernel_size: int = 3, border: float = 0.2) -> np.ndarray:
    """PCA foreground mask, copied line for line from AnomalyDINO `compute_background_mask`.

    Note: AnomalyDINO dilates/closes the mask while it still has shape (P, 1), i.e. along
    the flattened raster order, not on the 2-D grid. We keep that quirk on purpose so the
    reproduced baseline matches their numbers. Returns a bool array of shape (P,).
    """
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    first_pc = pca.fit_transform(features.astype(np.float32))          # (P, 1)
    mask = first_pc > threshold
    gh, gw = grid
    centre = mask.reshape(grid)[int(gh * border):int(gh * (1 - border)),
                                int(gw * border):int(gw * (1 - border))]
    if centre.sum() <= centre.size * 0.35:
        mask = -first_pc > threshold
    k = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return mask.squeeze()
