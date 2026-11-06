# Algorithm

1. Preserve source pixels/range metadata; produce grayscale and float32 `[0,1]` images with optional CLAHE/Gaussian preprocessing.
2. Extract global SIFT, compute SLIC, extract masked SIFT per region, and merge configurable duplicates.
3. BF-L2 KNN matching and Lowe filtering; deterministic four-point RANSAC and normalized DLT yield target-to-reference `H`.
4. Compute real Gabor filter-bank energy and replaceable saliency features. Refine eight homography parameters using the enabled constraints and reject implausible transforms.
5. Transform both image corner sets, allocate their complete union canvas, translate both images, blend by linear distance weights, and crop only the union-mask bounding box.
6. Save every stage, metrics, checkerboards, status, and an 8×2 debug montage. Ablations disable region, Gabor, and/or saliency modules without replacing SIFT.
