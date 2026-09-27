"""End-to-end two-image paper reproduction pipeline."""

from pathlib import Path
import csv, json, logging, time
import cv2
import numpy as np
from .image import (
	load_image,
	to_grayscale,
	normalize_image,
	save_image,
	enhance_ultrasound,
)
from .features import (
	extract_global_sift,
	extract_region_sift,
	merge_features,
	save_feature_set,
)
from .features import compute_slic
from .features import match_features, match_statistics
from .registration import ransac_homography
from .registration import warp_to_union
from .constraints import build_gabor_filter_bank, apply_gabor_bank, compute_gabor_energy
from .constraints import compute_saliency, extract_salient_points, match_salient_points
from .constraints import refine_homography
from .output import linear_blend, crop_to_mask
from .output import evaluate_pair
from .output import (
	draw_slic,
	draw_features,
	draw_matches,
	overlay,
	checkerboard,
	montage,
)

LOG = logging.getLogger(__name__)
MODES = {
	"sift": (False, False, False),
	"sift_slic": (True, False, False),
	"sift_slic_gabor": (True, True, False),
	"sift_slic_saliency": (True, False, True),
	"full": (True, True, True),
}


class PipelineError(RuntimeError):
	def __init__(self, status, message):
		super().__init__(message)
		self.status = status


def run_pipeline(reference_path, target_path, output_dir, config, mode="full"):
	start = time.time()
	out = Path(output_dir)
	[
		(out / f"{i:02d}_{n}").mkdir(parents=True, exist_ok=True)
		for i, n in enumerate(
			[
				"preprocessed",
				"slic",
				"sift",
				"matches",
				"initial_homography",
				"gabor",
				"saliency",
				"refinement",
				"warp",
				"fusion",
			],
			1,
		)
	]
	use_slic, use_gabor, use_saliency = MODES[mode]
	try:
		lr, lt = load_image(reference_path), load_image(target_path)
		rg, tg = to_grayscale(lr.original), to_grayscale(lt.original)
		rn, tn = normalize_image(rg), normalize_image(tg)
		re, te = enhance_ultrasound(rg, config["preprocessing"]), enhance_ultrasound(
			tg, config["preprocessing"]
		)
		LOG.info("Image size reference=%s target=%s", rg.shape, tg.shape)
		for name, x in [
			("reference_gray", normalize_image(rg)),
			("target_gray", normalize_image(tg)),
			("reference_normalized", rn),
			("target_normalized", tn),
			("reference_enhanced", re),
			("target_enhanced", te),
		]:
			save_image(out / "01_preprocessed" / f"{name}.png", x)
		rglob, tglob = extract_global_sift(re, config["sift"]), extract_global_sift(
			te, config["sift"]
		)
		rl, rb = compute_slic(re, config["slic"])
		tl, tb = compute_slic(te, config["slic"])
		np.save(out / "02_slic/reference_labels.npy", rl)
		np.save(out / "02_slic/target_labels.npy", tl)
		save_image(out / "02_slic/reference_slic.png", draw_slic(re, rl))
		save_image(out / "02_slic/target_slic.png", draw_slic(te, tl))
		rreg = (
			extract_region_sift(re, rl, config["sift"]) if use_slic else type(rglob)([])
		)
		treg = (
			extract_region_sift(te, tl, config["sift"]) if use_slic else type(tglob)([])
		)
		rf = merge_features(
			rglob,
			rreg,
			config["features"]["duplicate_radius"],
			config["features"]["descriptor_cosine_threshold"],
		)
		tf = merge_features(
			tglob,
			treg,
			config["features"]["duplicate_radius"],
			config["features"]["descriptor_cosine_threshold"],
		)
		LOG.info(
			"SLIC count ref=%d target=%d | Global SIFT ref=%d target=%d | Region SIFT ref=%d target=%d | After merging ref=%d target=%d",
			len(np.unique(rl)),
			len(np.unique(tl)),
			len(rglob.features),
			len(tglob.features),
			len(rreg.features),
			len(treg.features),
			len(rf.features),
			len(tf.features),
		)
		if min(len(rf.features), len(tf.features)) < config["sift"]["min_keypoints"]:
			raise PipelineError("FAILED_FEATURE", "Too few SIFT keypoints")
		for name, img, fs in [
			("reference_sift", re, rglob),
			("target_sift", te, tglob),
			("reference_region_sift", re, rreg),
			("target_region_sift", te, treg),
			("reference_all_features", re, rf),
			("target_all_features", te, tf),
		]:
			save_image(out / "03_sift" / f"{name}.png", draw_features(img, fs))
		save_feature_set(str(out / "03_sift/reference_keypoints.npz"), rf)
		save_feature_set(str(out / "03_sift/target_keypoints.npz"), tf)
		raw, good = match_features(rf, tf, config["matching"])
		LOG.info(
			"Raw matches=%d ratio matches=%d stats=%s",
			len(raw),
			len(good),
			match_statistics(good),
		)
		save_image(
			out / "04_matches/matches_before_filter.png",
			draw_matches(re, te, rf, tf, raw),
		)
		save_image(
			out / "04_matches/matches_after_ratio.png",
			draw_matches(re, te, rf, tf, good),
		)
		if len(good) < 4:
			raise PipelineError("FAILED_MATCH", f"Only {len(good)} matches")
		rp = np.array(
			[
				[rf.features[m.reference_index].x, rf.features[m.reference_index].y]
				for m in good
			]
		)
		tp = np.array(
			[
				[tf.features[m.target_index].x, tf.features[m.target_index].y]
				for m in good
			]
		)
		rr = ransac_homography(tp, rp, config["ransac"])
		H0 = rr.H
		np.save(out / "05_initial_homography/H_initial.npy", H0)
		inlier_matches = [m for m, k in zip(good, rr.inliers) if k]
		save_image(
			out / "05_initial_homography/homography_inliers.png",
			draw_matches(re, te, rf, tf, inlier_matches),
		)
		LOG.info(
			"RANSAC inliers=%d ratio=%.3f reprojection=%.4f",
			rr.inliers.sum(),
			rr.inliers.mean(),
			rr.errors[rr.inliers].mean(),
		)
		if rr.inliers.sum() < max(4, int(config["ransac"]["min_inliers"])):
			raise PipelineError("FAILED_H", f"Only {rr.inliers.sum()} RANSAC inliers")
		kernels = build_gabor_filter_bank(config["gabor"])
		rresp = apply_gabor_bank(re, kernels)
		tresp = apply_gabor_bank(te, kernels)
		renergy = compute_gabor_energy(
			rresp,
			config["gabor"]["energy_window"],
			config["gabor"]["downsample_factor"],
		)
		tenergy = compute_gabor_energy(
			tresp,
			config["gabor"]["energy_window"],
			config["gabor"]["downsample_factor"],
		)
		for i, x in enumerate(rresp):
			save_image(
				out
				/ "06_gabor"
				/ f'gabor_v{i//config["gabor"]["orientations"]}_u{i%config["gabor"]["orientations"]}.png',
				np.abs(x) / (np.max(np.abs(x)) + 1e-9),
			)
		gemap = np.sqrt(np.sum(renergy, axis=0))
		save_image(out / "06_gabor/gabor_energy_map.png", gemap / (gemap.max() + 1e-9))
		rs = compute_saliency(re, config["saliency"]["method"])
		ts = compute_saliency(te, config["saliency"]["method"])
		save_image(out / "07_saliency/reference_saliency.png", rs)
		save_image(out / "07_saliency/target_saliency.png", ts)
		rsp = extract_salient_points(
			rs,
			config["saliency"]["percentile"],
			config["saliency"]["max_points"],
			config["saliency"]["min_distance"],
		)
		tsp = extract_salient_points(
			ts,
			config["saliency"]["percentile"],
			config["saliency"]["max_points"],
			config["saliency"]["min_distance"],
		)
		sr, st = match_salient_points(
			rsp, tsp, H0, config["saliency"]["match_distance"]
		)
		irp = rp[rr.inliers]
		itp = tp[rr.inliers]
		ref = (
			refine_homography(
				H0,
				renergy,
				tenergy,
				irp,
				itp,
				sr,
				st,
				config["refinement"],
				config["transform_check"],
				te.shape,
				use_gabor,
				use_saliency,
			)
			if (use_gabor or use_saliency)
			else None
		)
		H = ref.H if ref else H0
		np.save(out / "08_refinement/H_refined.npy", H)
		LOG.info(
			"Eg before/after=%s/%s Es before/after=%s/%s",
			ref.Eg_before if ref else 0,
			ref.Eg_after if ref else 0,
			ref.Es_before if ref else 0,
			ref.Es_after if ref else 0,
		)
		wi = warp_to_union(rn, tn, H, config["transform_check"]["max_canvas_pixels"])
		union = wi.reference_mask | wi.target_mask
		overlap = wi.reference_mask & wi.target_mask
		if not overlap.any():
			raise PipelineError("NO_OVERLAP", "Warped images do not overlap")
		fused, weights = linear_blend(
			wi.reference,
			wi.target,
			wi.reference_mask,
			wi.target_mask,
			config["blending"]["method"],
		)
		cropped = crop_to_mask(fused, union)
		for name, x in [
			("prealigned_reference", wi.reference),
			("prealigned_target", wi.target),
			("prealigned_overlay", overlay(wi.reference, wi.target)),
		]:
			save_image(out / "09_warp" / f"{name}.png", x)
		for name, x in [
			("reference_mask", wi.reference_mask),
			("target_mask", wi.target_mask),
			("overlap_mask", overlap),
			("stitched_raw", fused),
			("stitched_cropped", cropped),
		]:
			save_image(out / "10_fusion" / f"{name}.png", x.astype(np.float32))
		save_image(out / "final_stitched.png", cropped)
		save_image(
			out / "checkerboard_initial.png", checkerboard(wi.reference, wi.target)
		)
		save_image(
			out / "checkerboard_refined.png", checkerboard(wi.reference, wi.target)
		)
		metrics = {
			r: evaluate_pair(wi.reference, wi.target, fused, m)
			for r, m in {
				"full": np.ones_like(union),
				"overlap": overlap,
				"union": union,
				"seam_band": cv2.dilate(
					overlap.astype(np.uint8), np.ones((5, 5), np.uint8)
				).astype(bool)
				& union,
			}.items()
		}
		(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
		with (out / "metrics.csv").open("w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["region", "mse", "psnr", "ssim", "ncc"])
			[w.writerow([k, *v.values()]) for k, v in metrics.items()]
		vis = [
			("reference", rn),
			("target", tn),
			("reference SLIC", draw_slic(re, rl)),
			("target SLIC", draw_slic(te, tl)),
			("reference SIFT", draw_features(re, rf)),
			("target SIFT", draw_features(te, tf)),
			("raw matches", draw_matches(re, te, rf, tf, raw)),
			("RANSAC inliers", draw_matches(re, te, rf, tf, inlier_matches)),
			("prealigned reference", wi.reference),
			("prealigned target", wi.target),
			("Gabor energy", gemap / (gemap.max() + 1e-9)),
			("saliency", rs),
			("initial overlay", overlay(wi.reference, wi.target)),
			("refined overlay", overlay(wi.reference, wi.target)),
			("overlap mask", overlap),
			("fusion result", fused),
		]
		montage(vis, out / "pipeline_debug.png")
		status = {
			"status": "SUCCESS",
			"runtime_seconds": time.time() - start,
			"canvas_size": list(fused.shape[:2][::-1]),
			"overlap_pixels": int(overlap.sum()),
			"mode": mode,
		}
		(out / "status.json").write_text(json.dumps(status, indent=2))
		LOG.info(
			"canvas=%s overlap=%d runtime=%.2fs",
			fused.shape,
			overlap.sum(),
			status["runtime_seconds"],
		)
		return status
	except PipelineError as e:
		status = {
			"status": e.status,
			"message": str(e),
			"runtime_seconds": time.time() - start,
		}
		(out / "status.json").write_text(json.dumps(status, indent=2))
		LOG.warning("%s: %s", e.status, e)
		return status
