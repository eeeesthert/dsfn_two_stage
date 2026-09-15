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

```bash
python scripts/run_dataset.py --manifest pairs.csv --output outputs/batch
python scripts/evaluate.py aligned_reference.png aligned_target.png
```

The manifest has `case,reference,target` columns. A failed case produces a status record and does not terminate the batch. Metrics include paper MSE/PSNR/SSIM plus clearly identified engineering-extension NCC and full/overlap/union/seam-band regions.

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
