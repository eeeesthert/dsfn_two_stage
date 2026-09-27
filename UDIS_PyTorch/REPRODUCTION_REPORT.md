# Reproduction report

## Delivered files

`configs/` contains the paper defaults. Related manifest, synthetic, aligned, and ABUS readers are consolidated in `datasets/data.py`. `models/alignment.py` contains the complete Stage-1 execution path from feature extraction through stitching-domain transformation. `models/reconstruction.py` contains both reconstruction branches and the frozen VGG extractor. `models/udis.py` is the small public composition API. `losses/` contains both stage objectives. Image utilities are in `utils/image.py`, while checkpoint and seed helpers are consolidated in `utils/runtime.py`. Root scripts train the stages separately, materialize aligned data, infer, and evaluate ABUS results.

`infer_abus.py` reads native `case/input1,input2,input3` slice trees and writes stage 12 and stage 23 results. `evaluate_abus.py` reports overlap MSE, PSNR, NCC, and valid-pixel counts without requiring ground-truth stitched images.

## Tensor flow

Stage 1: RGB `B×3×H×W → B×1×128×128 → F4 128×16² → CV 1089×16² → ΔP1`. warp `F3` and produce `CV 289×32² → ΔP2`. warp `F2` and produce `CV 81×64² → ΔP3`. sum eight offsets, rescale x/y, DLT `B×3×3`, and union-warp two RGB images plus two RGB content masks.

Stage 2: frozen warps `B×6×H×W → resize 256² → U-Net → SLR B×3×256²`. concatenate full-resolution pair plus resized SLR `B×9×H×W → 64 channels → eight residual blocks/global skip → SHR B×3×H×W`.

## Exact objectives and conventions

Alignment weights are `16:4:1`. Both LR and HR use `2·seam + 10⁻⁶·content`. final weights are `100:1:1` for LR, HR, consistency. Seam masks use the *other image's* dilated boundary. Full formulas are in README and executable loss modules.

Offsets define source corner → displaced target corner. DLT returns forward source→target H. `grid_sample` requires output→input coordinates. the warp therefore computes H inverse. Pixel normalization is explicit. Canvas minima may be negative and are retained as an origin. dimensions round upward to multiples of eight without cropping.

## Validation and uncertainty

Pytest validates identity/translation/gradient DLT, all cost-volume channel counts and gradients, forward translation direction, positive/negative union canvases, feature shapes, and reconstruction output/backward. ABUS tests verify name-based slice pairing, unmodified dimensions, output discovery, and overlap evaluation. The exact published FC layers make a full HomographyNet forward unusually RAM-intensive. The inspection tool instantiates and counts the exact model without retaining gradients. See README “Differences and numerical uncertainty” for TensorFlow/PyTorch resize, initialization, VGG, batching, and memory differences that can affect numerical—not architectural—reproduction.

Native ABUS training defaults to batch size one because cases may have different original dimensions.

## Running

```bash
cd UDIS_PyTorch
python train_alignment.py --config configs/alignment.yaml
python generate_aligned_dataset.py --checkpoint checkpoints/alignment_latest.pth --input pairs.csv --output aligned_dataset
python train_reconstruction.py --config configs/reconstruction.yaml
python infer.py --image1 a.jpg --image2 b.jpg --alignment_ckpt checkpoints/alignment_latest.pth --reconstruction_ckpt checkpoints/reconstruction_latest.pth --output result.jpg
python infer_abus.py --dataset-root ../dataset --alignment-ckpt checkpoints/alignment_latest.pth --reconstruction-ckpt checkpoints/reconstruction_latest.pth --out-dir ../outputs/udis_abus
python evaluate_abus.py --result-root ../outputs/udis_abus
```
