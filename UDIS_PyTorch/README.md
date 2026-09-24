# UDIS PyTorch reproduction

Faithful, two-stage PyTorch baseline for **Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images**, Nie et al., IEEE TIP 30 (2021), 6184–6197. This directory is intentionally isolated from the repository's pre-existing ABUS/DSFN code.

## Audit and scope

The repository originally contained an ABUS-specific pairwise pipeline and a vendored UDIS++ (2023) contract, but no implementation of the original 2021 UDIS. This directory adds the missing original feature pyramid, correlation, DLT, homography canvas, reconstruction, losses, datasets, training, inference, tests, and inspection tooling without changing ABUS modules.

## Method

### Stage 1 — coarse alignment

Both RGB inputs are bilinearly resized to 128² (`align_corners=False`) and independently converted by channel mean to grayscale. A shared, normalization-free pyramid produces `F1 64×128²`, `F2 64×64²`, `F3 128×32²`, and `F4 128×16²`. All Conv/FC biases are zero and weights use Xavier uniform.

The true correlation volume first channel-L2-normalizes both features, then enumerates `(dy,dx)` shifts and averages channel products. Ranges 16, 8, 4 yield 1089, 289, 81 channels. The regressors are:

* L1: `1089→512→512→512`, flatten, `→1024→8`.
* L2: `289→256→256→256(stride 2)`, flatten, `→512→8`.
* L3: `81→128→128(stride 2)→128(stride 2)`, flatten, `→256→8`.

Each hidden layer uses ReLU. FC uses dropout 0.5. L1 warps F3 with `DLT(delta1/4, 32)`. L2 warps F2 with `DLT((delta1+delta2)/2, 64)`. The final prediction is exactly `delta1+delta2+delta3`, eight values representing four `(dx,dy)` pairs.

**Coordinates.** DLT corners are TL, TR, BL, BR. `DifferentiableDLT` solves `A h=b` with `torch.linalg.solve` and returns the forward mapping **source → target**, where target corners are `P+delta`. `grid_sample` is backward sampling, so the warp explicitly applies `H⁻¹` to every output target pixel, then converts source pixels to normalized coordinates with `align_corners=True`. Resize coordinates are separate and consistently use `align_corners=False`.

For original `H×W`, x/y offsets are respectively scaled by `W/128`, `H/128`, and DLT is recomputed. The stitching transformer takes the union of reference corners and transformed source corners, preserves negative minima through an explicit canvas origin, and rounds the canvas upward to multiples of eight. Images and all-one tensors pass through identical transforms to produce RGB masks in `[0,1]`.

The training loss is

`Lalign = 16 |warp(I2,H1)-I1·M1|₁ + 4 |warp(I2,H2)-I1·M2|₁ + |warp(I2,H3)-I1·M3|₁`.

Independent brightness and RGB gains are sampled in `[0.7,1.3]` only for network inputs. unaugmented images form the loss. Adam starts at `1e-4`, exponential decay is `0.96` per 12,500 steps, gradient norm is clipped to 3, and default training is 600,000 iterations/batch 4.

### Stage 2 — reconstruction

Stage 2 reads frozen `warp1/warp2/mask1/mask2`. it never invokes Stage 1 and never uses a ground-truth stitch. The LR U-Net resizes inputs to 256², uses channels `6→64→128→256→512` with three max pools, transpose-convolution decoding and matching skips, and emits a tanh RGB image. The HR branch concatenates both full-resolution warps and upsampled LR output (`9` channels), uses `9→64`, eight normalization-free residual blocks, a global residual, and tanh RGB output.

For a mask `M`, boundary is the sum of vertical/horizontal absolute neighbor differences, saturated after three 3×3 all-one convolutions. Cross masks are `seam1=dilate(boundary(M2))·M1` and `seam2=dilate(boundary(M1))·M2` (not overlap masks).

Frozen ImageNet VGG19 features are index 14 (`conv3_3`, after its ReLU at 15 is deliberately not selected) and index 32 (`conv5_3`). Inputs convert `[-1,1]→[0,1]` and use ImageNet mean/std. Masked images resize to 224². LR content uses `conv5_3`. HR content uses `conv3_3`:

* `LLR = 2·LRseam + 1e-6·LRcontent`
* `LHR = 2·HRseam + 1e-6·HRcontent`
* `Lconsistency = |resize(SHR,256²)-SLR|₁`
* `Ltotal = 100·LLR + LHR + Lconsistency`

Adam starts at `1e-4`, decay is `0.98` per 10,000 steps, batch is one, and default training is 200,000 iterations.

## Data

Stage-1 CSV may have header `image1,image2`, followed by paths. Synthetic pretraining accepts a directory of single images and returns random four-corner perturbations. known offsets are exposed only as debug metadata and training remains photometric.

Aligned data layout is `aligned_dataset/{training,testing}/{warp1,warp2,mask1,mask2}/same_name.png`. Images are RGB `float32` in `[-1,1]`. masks are `[0,1]`. Stage 2 proportionally limits dimensions to 1024 and aligns them to multiples of eight.

## Commands

```bash
cd UDIS_PyTorch
python train_alignment.py --config configs/alignment.yaml
python generate_aligned_dataset.py --checkpoint checkpoints/alignment_latest.pth --input pairs.csv --output aligned_dataset
python train_reconstruction.py --config configs/reconstruction.yaml
python infer.py --image1 a.jpg --image2 b.jpg --alignment_ckpt checkpoints/alignment_latest.pth --reconstruction_ckpt checkpoints/reconstruction_latest.pth --output stitch.jpg --save_intermediates
pytest -q
python tools/inspect_model.py
```

Both training scripts accept `--resume`. Checkpoints contain model, optimizer, scheduler, global step, and configuration.

## Differences and numerical uncertainty

This is a transparent PyTorch port, not UDIS++. TensorFlow resize is represented by PyTorch bilinear `align_corners=False`. explicit geometric sampling uses `align_corners=True` to preserve integer endpoint pixel coordinates. PyTorch Xavier random streams, dropout, padding, and VGG preprocessing can therefore differ numerically from historical TensorFlow. Dynamic canvases must be rectangular for a tensor batch, so a batch uses the union bounds across its samples. published Stage-2 batch size is one. VGG weights are downloaded by torchvision on first use. The exact L1/FC architecture is intentionally retained and the alignment network is memory-heavy (roughly 1 GB including gradients).

## Official implementation reference

The implementation follows the paper and the authors' repository at
[nie-lang/UnsupervisedDeepImageStitching](https://github.com/nie-lang/UnsupervisedDeepImageStitching).
The port keeps the official two-stage separation, four-corner parameterization,
coarse-to-fine correlation regressors, stitching-domain output, and
reconstruction losses. TensorFlow and PyTorch sampling differences are listed
above instead of being hidden behind an alternative design.

## Native ABUS input

ABUS data can be read directly. No manifest conversion, crop, or original-image
resize is performed. Paired inputs must have equal dimensions.

```text
dataset/
└── case001/
    ├── input1/slice_0001.jpg
    ├── input2/slice_0001.jpg
    └── input3/slice_0001.jpg
```

A single image such as `case001/input1.jpg` is also accepted. Stage `12` pairs
input1 with input2. Stage `23` pairs input2 with input3. Matching slice names
are used first, with sorted positional pairing only as a compatibility fallback.

Train Stage 1 directly from ABUS input:

```bash
python UDIS_PyTorch/train_alignment.py \
    --config UDIS_PyTorch/configs/alignment.yaml \
    --dataset-root ./dataset \
    --stages 12 23
```

### Complete ABUS training order without CSV

You do **not** need to create a CSV for the repository's native ABUS layout,
and you do not need to edit either YAML file when using the default paths.
Run commands from the repository root in this order:

1. Train the shared Stage-1 alignment model directly from both ABUS pair types.

   ```bash
   python UDIS_PyTorch/train_alignment.py \
       --config UDIS_PyTorch/configs/alignment.yaml \
       --dataset-root ./dataset \
       --stages 12 23
   ```

2. Use the trained alignment checkpoint to generate the frozen Stage-2 dataset.

   ```bash
   python UDIS_PyTorch/generate_aligned_dataset.py \
       --checkpoint checkpoints/alignment_latest.pth \
       --dataset-root ./dataset \
       --stages 12 23 \
       --output aligned_dataset \
       --split training
   ```

3. Train Stage 2. In the paper and this project it is called
   **reconstruction**, not registration. The default reconstruction config
   already reads `aligned_dataset/training`.

   ```bash
   python UDIS_PyTorch/train_reconstruction.py \
       --config UDIS_PyTorch/configs/reconstruction.yaml
   ```

4. Run inference after both checkpoints exist.

   ```bash
   python UDIS_PyTorch/infer_abus.py \
       --dataset-root ./dataset \
       --alignment-ckpt checkpoints/alignment_latest.pth \
       --reconstruction-ckpt checkpoints/reconstruction_latest.pth \
       --out-dir outputs/udis_abus \
       --stages 12 23
   ```

The CSV path remains available only for non-ABUS datasets. Use either
`--input pairs.csv` or `--dataset-root ./dataset`, never both.

Run both pairwise stages and write files compatible with the repository's
`stage/case/fusion` convention:

```bash
python UDIS_PyTorch/infer_abus.py \
    --dataset-root ./dataset \
    --alignment-ckpt ./checkpoints/alignment_latest.pth \
    --reconstruction-ckpt ./checkpoints/reconstruction_latest.pth \
    --out-dir ./outputs/udis_abus \
    --stages 12 23
```

Evaluate registration in the valid overlap without requiring a ground-truth
stitched image:

```bash
python UDIS_PyTorch/evaluate_abus.py \
    --result-root ./outputs/udis_abus \
    --out-csv ./outputs/udis_abus/alignment_metrics.csv
```

The CSV contains per-case, per-slice overlap MSE, PSNR, NCC, and valid-pixel
count. The inference output also uses names such as
`12_0001_stitched.png`, `12_0001_mask_left_soft.png`, and
`12_0001_mask_right_soft.png`, so downstream ABUS tools can discover it.

## Source organization and formatting

Related dataset implementations now live together in `datasets/data.py`.
All Stage-1 model components now live in `models/alignment.py`, in execution
order: initialization, cost volume, DLT, warp, feature pyramid, regressors,
stitching transformer, and alignment pipeline. The reconstruction network and
VGG extractor live together in `models/reconstruction.py`. This replaces nine
small model files with two stage-oriented modules without changing their APIs.
Checkpoint and random-seed helpers live in `utils/runtime.py`. The unused image
logging helper was removed.
The duplicate alignment-export command was removed in favor of the single
`generate_aligned_dataset.py` entry point. Python blocks use tabs consistently,
and statements are placed on separate lines rather than joined by semicolons.

The tests are not runtime dependencies. They protect the error-prone geometry,
tensor shapes, gradients, ABUS pairing, and reconstruction output. The separate
Stage-1 test files were therefore merged into `tests/test_alignment.py`. The
disabled full-network smoke test was removed because it duplicated the model
inspection script while being skipped in normal test runs.
