# DunHuangStitch — PyTorch reproduction

## Paper
Reproduction of **DunHuangStitch: Unsupervised Deep Image Stitching of Dunhuang Murals**, Yuan Mei et al., IEEE TVCG (2025). This is an independent implementation from the paper description, not official author code.

## Method overview and architecture
The pipeline is deliberately **not jointly trained**:

1. A shared extractor (Focus, six parallel HABs, three PatchMerging operations) produces `[coarse, middle, fine]` pyramids.
2. Alignment stages use symmetric-radius correlation cost volume → displacement-channel attention → APRB compression → convolutional HPB. Residual four-corner predictions are converted to full-resolution pixels and cumulatively added. Normalized differentiable DLT produces three global target→reference homographies. `grid_sample` produces images and valid masks.
3. The frozen alignment checkpoint generates the aligned dataset.
4. Fusion applies a shared three-RepConv shallow encoder, signed feature difference, four-down/four-up reconstruction U-Net with **summation** skips, a self-attention bottleneck, and a two-upsample/two-RepConv sigmoid seam generator. The continuous seam weights valid aligned images; it is never thresholded in training.

At 128×128 with defaults, extractor features are fine `[B,48,32,32]`, middle `[B,64,16,16]`, coarse `[B,64,8,8]`; stages consume coarse/middle/fine, APRB emits `[B,49,h,w]`, each delta/offset is `[B,4,2]`, each H is `[B,3,3]`, and training warps/masks are `[B,3,128,128]`/`[B,1,128,128]`. Fusion shallow/difference features are `[B,24,H,W]`; configured U-Net widths are 24/32/48/64/96 and output seam is `[B,1,H,W]`.

## Environment
Python 3.10+, PyTorch 2.x. Install with `pip install -r requirements.txt`. Training geometry is pure PyTorch: no OpenCV/NumPy DLT or warp occurs in the gradient path.

## Dataset format
Original pairs: `ROOT/{train,test}/case_x/reference.png,target.png`, or pass a two-column pair list to `ImagePairDataset`. Aligned pairs additionally contain `mask_reference.png`, `mask_target.png`, and `homography.npy`.

## Synthetic pair generation
`SyntheticHomographyDataset` loads an image, resizes/crops to the configured resolution, perturbs four corners, solves DLT, and returns reference, target, `H_gt`, and `offset_gt`. Ground truth is debug/evaluation-only; optimization remains photometric and unsupervised. Perturbation magnitude is configurable because the paper does not explicitly specify it.

## Train alignment
`python train_alignment.py --config configs/alignment.yaml --output checkpoints/alignment [--resume .../last.pt]`

## Generate aligned dataset
`python generate_aligned_dataset.py --checkpoint checkpoints/alignment/best.pt --input_root data --output_root aligned_dataset --split train --device cuda`

## Train fusion
`python train_fusion.py --config configs/fusion.yaml --output checkpoints/fusion [--resume .../last.pt]`

## Inference
`python inference.py --reference ref.jpg --target tgt.jpg --alignment_checkpoint checkpoints/alignment/best.pt --fusion_checkpoint checkpoints/fusion/best.pt --output outputs/result.png`

Inference computes the union of reference corners and H-transformed target corners, applies a positive-canvas translation, warps both images/masks, then fuses. Union-canvas handling is a necessary engineering implementation detail not specified by the paper. Debug warps, masks, seams and result are saved beside the output.

## Evaluation and visualization
`metrics/alignment_metrics.py` supplies RMSE, PSNR, and SSIM; synthetic offsets permit four-point RMSE. `metrics/seam_quality.py` supplies lower-is-better `Q_seam = 1000 E_patch + E_point`, using seam dilation radius 5. Debug writers save every alignment stage and fusion seam. `evaluate.py` exposes the metric API for dataset-specific evaluation.

## Reparameterization / deploy
Call `switch_to_deploy()` on each `RepConv` after `eval()`. It fuses 3×3+BN, 1×1+BN and eligible identity+BN branches into one 3×3 convolution. Unit tests check equivalence.

## Unit tests and smoke test
`pytest tests/` validates DLT mappings/gradient, warper, cost volume/backward, RepConv deploy, continuous fusion, weighted validity regions, and seam-loss backward. `python tests/smoke_test.py` runs alignment optimization, aligned-pair handoff, fusion and loss backward on CPU.

## Paper-specified hyperparameters
| Item | Paper value |
|---|---|
| Alignment stages / homography | 3 / global-global-global |
| Search ranges | 16 / 8 / 4 |
| APRB output | 49 channels |
| HAB channel perception | depthwise 7×7 |
| Activation | GELU |
| Alignment L1 weights | 1 / 4 / 16 |
| Alignment optimizer / epochs / LR / warmup | AdamW / 200 / 1e-4 / 20 |
| Fusion optimizer / epochs / LR / warmup | AdamW / 50 / 1e-4 / 5 |
| Synthetic resolution / overlap | 128×128 / 20%–100% |
| Paper train / test images | 8051 / 1714 |
| Real test pairs / resolution | 35 / 624×936 |
| Seam-quality gamma | 1000 |

## Paper-unspecified implementation choices
**以下参数为工程复现假设，不是论文明确报告值 (These values are engineering defaults, not values explicitly reported in the paper).** Every choice is in YAML: feature widths; heads; window size; MLP ratio; QKV/library attention details; HAB projection/residual; symmetric cost-volume interpretation of “range”; APRB attention reduction and convolution count; HPB widths/count; batch size; AdamW betas/weight decay; exponential gamma; fusion widths; signed rather than absolute feature subtraction; interpolation method; edge alpha; perturbation rho; mask-loss normalization epsilon; and RepConv detailed widths. `paper_literal` implements the printed signed neighboring fused-image difference in Lf; `stable_abs` is an explicit optional experiment. The paper does not explicitly specify these parameters.

## Remaining ambiguities
The official displacement-window convention, exact HAB/HPB/APRB microarchitecture, feature-difference sign/absolute choice, FastViT modification details, seam edge normalization, Lf missing absolute sign, alpha, and synthetic perturbation distribution cannot be uniquely recovered from the article description. Config/module boundaries isolate each assumption for replacement if official code becomes available.
