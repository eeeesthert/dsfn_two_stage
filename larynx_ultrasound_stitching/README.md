# Larynx ultrasound image stitching

Independent Python reproduction of Yadi Yan et al., **“Larynx Ultrasound Image Stitching Based on Multiconstraint Super-Pixel Feature,”** IEEE TIM 72 (2023), DOI `10.1109/TIM.2023.3243675`.

> This repository is an independent reproduction based on the methodology described in the paper and is not the authors' official implementation.

## Algorithm overview

The pipeline follows preprocessing → global SIFT → SLIC → masked region SIFT → merged descriptor matching → custom normalized DLT/RANSAC → point-region homography → 30 real Gabor responses/local energy → saliency points/variance → multi-constraint homography refinement → complete union canvas → linear weighted smoothing. It is deliberately not a SIFT/findHomography/average shortcut and uses no deep model.

## Environment installation

```bash
cd larynx_ultrasound_stitching
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is supported. Inputs may be PNG, JPEG, BMP, or TIFF; grayscale uint8/uint16 and BGR/BGRA screenshots are accepted. The original image and dynamic-range metadata remain untouched while feature processing uses float32 `[0,1]`.

## Single pair inference

```bash
python main.py --reference data/reference.png --target data/target.png --output outputs/case001
```

Use `--config configs/debug.yaml`, `--slic-k 50|100|150|200|300|400|500`, or one of the modes below.

## Ablation

`--mode sift`, `sift_slic`, `sift_slic_gabor`, `sift_slic_saliency`, or `full` implements experiments A–E. SLIC segmentation remains visualized in every mode, while region features are only used where enabled.

## Batch inference and evaluation

The ABUS-derived two-dimensional dataset runner accepts this layout:

```text
dataset/case001/input1/slice_0001.jpg
dataset/case001/input2/slice_0001.jpg
dataset/case001/input3/slice_0001.jpg
dataset/case001/nipple_x.txt
```

For every common slice filename, pair `12` uses `input1` as reference and `input2` as target. Pair `23` uses `input3` as reference and the same `input2` image as target. Discover the jobs without running image processing first:

```bash
python scripts/run_dataset.py --dataset-root ./dataset --output outputs/abus --dry-run
```

Run all discovered pairs with:

```bash
python scripts/run_dataset.py --dataset-root ./dataset --output outputs/abus --mode full
python scripts/evaluate.py aligned_reference.png aligned_target.png
```

The runner writes `jobs.json` before processing and `batch_status.json` afterwards. `nipple_x.txt` is recorded in each job as case metadata but is not used to alter the paper's registration objective. A failed slice does not terminate the remaining jobs. Metrics include paper MSE/PSNR/SSIM plus clearly identified engineering-extension NCC and full/overlap/union/seam-band regions.

## Intermediate results

Every run creates `01_preprocessed` through `10_fusion`, `final_stitched.png`, `pipeline_debug.png`, checkerboards, `metrics.json`, `metrics.csv`, and `status.json`. Keypoint NPZ files include coordinates, size, angle, response, octave, descriptor, source type, and super-pixel ID.

## Known differences from original implementation

The publication does not provide enough information to reproduce several choices exactly. In particular its saliency detector, feature correspondence details, Gabor energy reduction, sparse least-squares parameterization/Jacobian, and fusion weight equation are unavailable. Configurable and explicitly marked reproduction assumptions are used rather than invented paper claims. See [reproduction notes](docs/reproduction_notes.md), [algorithm](docs/algorithm.md), and [equations](docs/equations.md).

## Source organization

The implementation is intentionally grouped by responsibility instead of splitting every small operation into a separate module:

* `src/image.py` contains image I/O, normalization, grayscale conversion, and preprocessing.
* `src/features.py` contains SIFT, SLIC region extraction, feature merging, and matching.
* `src/registration.py` contains normalized DLT, RANSAC, canvas construction, and warping.
* `src/constraints.py` contains Gabor, saliency, transform validation, and refinement.
* `src/output.py` contains blending, metrics, and visualization.
* `src/pipeline.py` only orchestrates the complete workflow and its outputs.

Python blocks use tab indentation throughout this package. Statements are kept on separate lines rather than joined with semicolons.

## Inspecting batch results

The batch runner writes these indexes beside the per-case output directories:

* `summary.json` gives total, successful, failed, and per-status counts.
* `results.csv` contains every job with match and RANSAC diagnostics.
* `successful_outputs.txt` lists output directories that contain completed results.
* `failed_jobs.csv` contains only failed jobs and their reason.
* `batch_status.json` retains the complete machine-readable records.

A successful output directory contains `final_stitched.png` and `status.json` with status `SUCCESS`. The paper specifies four-point RANSAC sampling but does not publish a stricter final-inlier cutoff. Consequently, `ransac.min_inliers` defaults to four; increase it when a dataset requires more conservative acceptance.

## No-ground-truth batch evaluation

Completed ABUS-derived jobs can be evaluated with:

```bash
python scripts/evaluate_dataset.py \
	--output-root outputs/abus \
	--out-csv outputs/abus/evaluation.csv \
	--band-width 5
```

The evaluator reads each successful slice's `09_warp` and `10_fusion` outputs. It reports three comparisons separately: warped reference versus warped target (`registration`), fused image versus warped reference (`fused_reference`), and fused image versus warped target (`fused_target`). It never averages the two fusion sides into an unlabeled summary. Regions are restricted to valid support, overlap, and an overlap-boundary seam band, so zero-padded canvas pixels are not counted as image agreement.

These MSE, PSNR, SSIM, and NCC values are no-ground-truth consistency measurements. They do not measure physical registration error and must not be presented as TRE or landmark accuracy. When landmarks or a known synthetic homography are available, use those as the primary registration evaluation.
