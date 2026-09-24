from __future__ import annotations

import argparse
import json

from pixelstitch_abus.dataset import PixelStitchABUSDataset
from pixelstitch_abus.homography import HomographyProvider
from pixelstitch_abus.inference import build_model, infer_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot official PixelStitch inference on 2-D ABUS slice pairs")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", default="./checkpoints/ckpt.pth")
    parser.add_argument("--homography-root")
    parser.add_argument("--homography-source", choices=sorted(HomographyProvider.MODES), default="precomputed")
    parser.add_argument("--out-dir", default="./outputs/pixelstitch")
    parser.add_argument("--stages", nargs="+", choices=("12", "23"), default=("12", "23"))
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--corr-kernel-size", type=int, default=13)
    parser.add_argument("--case")
    parser.add_argument("--slice-id")
    parser.add_argument("--auto-crop", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = build_model(args.checkpoint, args.device, args.iters, args.corr_kernel_size)
    provider = HomographyProvider(args.homography_source, args.homography_root)
    processed = 0
    for stage in args.stages:
        dataset = PixelStitchABUSDataset(args.dataset_root, stage, args.image_size, args.case, args.slice_id)
        print(f"stage{stage}: {len(dataset)} pair(s)")
        for sample in dataset:
            print(
                f"case={sample['case']} slice={sample['slice_id']} stage={stage} "
                f"shape={tuple(sample['image1'].shape)} x1={float(sample['x1']):.2f} x2={float(sample['x2']):.2f}"
            )
            metadata = infer_sample(model, sample, provider, args.out_dir, args.device, args.iters, args.auto_crop)
            print(json.dumps({k: metadata[k] for k in ("case", "slice_id", "homography_source", "homography_fallback", "output_shape")}))
            processed += 1
    if processed == 0:
        raise RuntimeError("No ABUS pairs matched the requested filters")
    print(f"Completed {processed} independent 2-D pair(s).")


if __name__ == "__main__":
    main()
