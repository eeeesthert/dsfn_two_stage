# Reproduction report

## Delivered files

`configs/` contains the paper defaults; `datasets/` contains manifest, synthetic, and aligned datasets; `models/` contains the shared pyramid, correlation, DLT, warp, three regressors, canvas transformer, two reconstruction branches, VGG extractor, Stage-1 pipeline, and combined inference API; `losses/` contains both stage objectives; `utils/` contains image, checkpoint, seed, and visualization utilities. Root scripts train the stages separately, materialize aligned data, and infer. `tests/` and `tools/inspect_model.py` provide geometry, gradient, shape, range, canvas, and parameter checks.

## Tensor flow

Stage 1: RGB `B×3×H×W → B×1×128×128 → F4 128×16² → CV 1089×16² → ΔP1`; warp `F3` and produce `CV 289×32² → ΔP2`; warp `F2` and produce `CV 81×64² → ΔP3`; sum eight offsets, rescale x/y, DLT `B×3×3`, and union-warp two RGB images plus two RGB content masks.

Stage 2: frozen warps `B×6×H×W → resize 256² → U-Net → SLR B×3×256²`; concatenate full-resolution pair plus resized SLR `B×9×H×W → 64 channels → eight residual blocks/global skip → SHR B×3×H×W`.

## Exact objectives and conventions

Alignment weights are `16:4:1`. Both LR and HR use `2·seam + 10⁻⁶·content`; final weights are `100:1:1` for LR, HR, consistency. Seam masks use the *other image's* dilated boundary. Full formulas are in README and executable loss modules.

Offsets define source corner → displaced target corner. DLT returns forward source→target H. `grid_sample` requires output→input coordinates; the warp therefore computes H inverse. Pixel normalization is explicit. Canvas minima may be negative and are retained as an origin; dimensions round upward to multiples of eight without cropping.

## Validation and uncertainty

Pytest validates identity/translation/gradient DLT, all cost-volume channel counts and gradients, forward translation direction, positive/negative union canvases, feature shapes, and reconstruction output/backward. The exact published FC layers make a full HomographyNet forward unusually RAM-intensive and it is opt-in in constrained CI; the inspection tool still instantiates and counts the exact model. See README “Differences and numerical uncertainty” for TensorFlow/PyTorch resize, initialization, VGG, batching, and memory differences that can affect numerical—not architectural—reproduction.

## Running

```bash
cd UDIS_PyTorch
python train_alignment.py --config configs/alignment.yaml
python generate_aligned_dataset.py --checkpoint checkpoints/alignment_latest.pth --input pairs.csv --output aligned_dataset
python train_reconstruction.py --config configs/reconstruction.yaml
python infer.py --image1 a.jpg --image2 b.jpg --alignment_ckpt checkpoints/alignment_latest.pth --reconstruction_ckpt checkpoints/reconstruction_latest.pth --output result.jpg
```
