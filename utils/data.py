"""Dataset and transform builders."""

from __future__ import annotations

import os
from typing import Tuple

from torchvision import datasets, transforms

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def build_transforms(img_size: int, train: bool = True, strong_aug: bool = False):
    """Train / eval transforms.

    Evaluation is deliberately deterministic: resize + tensor + normalise only.
    Never put random operations (rotation, blur, erasing) into the val/test
    pipeline, otherwise the reported accuracy is not reproducible.
    """
    if train:
        scale = (0.4, 1.0) if strong_aug else (0.6, 1.0)
        jitter = 0.35 if strong_aug else 0.25
        ops = [
            transforms.RandomResizedCrop(img_size, scale=scale),
            transforms.RandomRotation(20 if strong_aug else 12),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=jitter, contrast=jitter,
                                   saturation=jitter, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
        ]
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_datasets(root: str, img_size: int, strong_aug: bool = False
                   ) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """Load ``root/train|val|test`` ImageFolder splits."""
    missing = [s for s in ("train", "val", "test") if not os.path.isdir(os.path.join(root, s))]
    if missing:
        raise FileNotFoundError(
            f"missing split(s) {missing} under '{root}'. Expected root/train, root/val, root/test."
        )

    train_ds = datasets.ImageFolder(os.path.join(root, "train"),
                                    transform=build_transforms(img_size, True, strong_aug))
    val_ds = datasets.ImageFolder(os.path.join(root, "val"),
                                  transform=build_transforms(img_size, False))
    test_ds = datasets.ImageFolder(os.path.join(root, "test"),
                                   transform=build_transforms(img_size, False))
    return train_ds, val_ds, test_ds
