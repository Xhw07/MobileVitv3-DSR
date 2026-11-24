"""Report parameter count and GFLOPs for the four ablation variants.

    python scripts/measure_complexity.py --img-size 96
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model, VARIANTS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    print(f"{'variant':<10}{'params (M)':>12}{'GFLOPs':>10}")
    print("-" * 32)
    for variant in VARIANTS:
        model = build_model(variant, num_classes=args.num_classes,
                            patch_size=args.patch_size).to(device).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6

        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as counter:
                model(dummy)
            flops = counter.get_total_flops() / 1e9
        except Exception:  # pragma: no cover - older torch
            flops = float("nan")

        print(f"{variant:<10}{params:>12.3f}{flops:>10.3f}")


if __name__ == "__main__":
    main()
