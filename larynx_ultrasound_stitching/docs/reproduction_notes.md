# Reproduction notes

This is an independent engineering reproduction, not authors' code. Where the article is insufficiently detailed, **论文没有提供足够信息，因此这里属于工程复现假设。**

## 1. Explicitly specified by paper

The method uses SIFT point features; SLIC super-pixels with compactness `m=10` and spacing `Ns=sqrt(N/K)`; region and point features; RANSAC with four-correspondence samples; DLT and a global 3×3 homography; real Gabor texture constraints with five scales and six orientations; a saliency constraint; `E=Eg+Es`; a sparse linear least-squares solver; linear weighted smoothing; PSNR; and SSIM.

## 2. Not specified by paper

The paper does not disclose SLIC `K`; SIFT implementation settings; descriptor ratio threshold; RANSAC reprojection threshold/confidence/budget; numerical Gabor sigma or finite kernel support; exact joint spatial-frequency local-energy window; bicubic target dimension; saliency detector; salient correspondence strategy; exact sparse-LS variables/Jacobian; feature deduplication/relative weights; transform plausibility limits; or exact blending-weight equation.

## 3. Our reproduction assumptions

* `K=200` is an initial engineering value—not an original paper parameter. CLI `--slic-k` scans 50, 100, 150, 200, 300, 400, or 500.
* OpenCV SIFT defaults listed in YAML, BF-L2 KNN, Lowe ratio 0.75, optional mutual matching, spatial/cosine deduplication, and deterministic RANSAC thresholds are replaceable assumptions.
* Formula-mode real Gabor kernels use sigma 2, finite 31-pixel support, squared-response box energy over 9 pixels, and bicubic factor 4. OpenCV kernels are an alternate backend.
* Spectral residual saliency, percentile/NMS points, initial-H nearest-neighbor correspondence, and global Eq. (10) spread are assumptions. Gradient, local variance, and (when available) fine-grained backends are alternatives.
* Because the paper omits optimization variables and Jacobian, nonlinear local eight-parameter homography least squares is the stable default; a finite-difference linearization with SciPy sparse `lsqr` is also implemented. This is not represented as author-official code.
* Distance-transform linear weights are the default approximation to linear smoothing; intensity weighting is optional. Regional metrics beyond paper PSNR/SSIM are engineering extensions.

## ABUS-derived dataset adapter

The dataset adapter treats `input2` as the target for both generated jobs. Pair `12` uses `input1` as reference and pair `23` uses `input3` as reference. Matching is by identical slice filename. `nipple_x.txt` is retained in `jobs.json` and `batch_status.json` as case metadata. The paper does not define a nipple-coordinate term in the two-image objective, so the adapter does not inject this value into registration or blending.
