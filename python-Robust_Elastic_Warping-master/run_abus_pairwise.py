"""Offline ABUS manifest entry point for the local REW implementation.

No training, downloads, resizing, cropping or learned replacement matcher.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from comp_KR import comp_KR
from load_data import load_data
from mosaic_local_ori import mosaic_local_ori

REQUIRED = ('case', 'stage', 'slice', 'prefix', 'left', 'right', 'left_x', 'right_x')
OPTIONAL = ('left_mask', 'right_mask')


def safe_component(value, field):
    if not value or value in {'.', '..'} or '/' in value or '\\' in value:
        raise ValueError(f'Invalid {field}: {value!r}')
    return value


def read_manifest(path):
    """Validate all rows before creating outputs; keep the supplied prefix."""
    path = Path(path).resolve()
    rows, seen = [], set()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing manifest columns: {sorted(missing)}')
        for line, raw in enumerate(reader, 2):
            if None in raw or any(raw.get(k) is None for k in REQUIRED):
                raise ValueError(f'Malformed manifest row {line}')
            row = {k: raw[k].strip() for k in REQUIRED}
            row.update({k: raw[k].strip() for k in OPTIONAL if raw.get(k) and raw[k].strip()})
            for key in ('case', 'slice', 'prefix'):
                safe_component(row[key], key)
            if row['stage'] not in {'12', '23'}:
                raise ValueError(f'Invalid stage at row {line}: {row["stage"]}')
            identity = (row['stage'], row['case'], row['prefix'])
            if identity in seen:
                raise ValueError(f'Duplicate output prefix at row {line}: {identity}')
            seen.add(identity)
            for key in ('left', 'right', *OPTIONAL):
                if key not in row:
                    continue
                image_path = Path(row[key]).expanduser()
                # Relative manifest paths are relative to the manifest directory.
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                row[key] = str(image_path.resolve())
                if not image_path.is_file():
                    raise ValueError(f'Missing {key} image at row {line}: {image_path}')
            # Nipple coordinates are retained for traceability. Original REW
            # estimates alignment from SIFT matches and does not consume them.
            rows.append(row)
    if not rows:
        raise ValueError('Manifest contains no pairs')
    return rows


def align_pair(row, seed, max_canvas_pixels, *, estimator="opencv", reproj_threshold=3.0,
               max_iters=10000, min_inliers=6, diagnostics=None,
               matching_options=None, geometry_options=None):
    info = diagnostics if diagnostics is not None else {}
    matching_options = matching_options or {}
    geometry_options = geometry_options or {}
    info.update(processing_stage="read_images", homography_estimator=estimator,
                matching_options=matching_options, geometry_options=geometry_options)
    images = [cv.imread(row[k], cv.IMREAD_COLOR) for k in ('left', 'right')]
    if any(im is None for im in images):
        raise ValueError('OpenCV could not decode the input pair')
    random.seed(seed)
    cv.setRNGSeed(seed)
    info.update(left_shape=list(images[0].shape), right_shape=list(images[1].shape), processing_stage='sift')
    feature_masks = []
    for key in OPTIONAL:
        mask = cv.imread(row[key], cv.IMREAD_GRAYSCALE) if row.get(key) else None
        if row.get(key) and mask is None:
            raise ValueError(f'Cannot decode feature mask: {row[key]}')
        feature_masks.append(mask)
    x1, x2 = load_data(*images, mask1=feature_masks[0], mask2=feature_masks[1],
                       diagnostics=info, **matching_options)
    info.update(matches=int(x1.shape[1]), processing_stage='homography')
    h, in1, in2 = comp_KR(*images, x1, x2, display=False, estimator=estimator,
                         reproj_threshold=reproj_threshold, max_iters=max_iters,
                         min_inliers=min_inliers, diagnostics=info, **geometry_options)
    info['processing_stage'] = 'canvas_or_tps'
    result = mosaic_local_ori(*images, h, in1, in2, display=False,
                              return_details=True, max_canvas_pixels=max_canvas_pixels)
    info['processing_stage'] = 'save_outputs'
    return result, info


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        raise OSError(f'Failed to save {path}')


def result_files(root, row):
    directory = root / row['stage'] / row['case']
    prefix = row['prefix']
    return {
        'left': directory / 'warp' / f'{prefix}_left.png',
        'right': directory / 'warp' / f'{prefix}_right.png',
        **{key: directory / 'fusion' / f'{prefix}_{key}.png' for key in (
            'stitched', 'mask_left_soft', 'mask_right_soft',
            'mask_left_bin', 'mask_right_bin', 'seam_soft',
            'valid_left', 'valid_right')},
        'metadata': directory / 'warp' / f'{prefix}_transform.json',
    }


def save_result(files, row, result, info):
    mosaic, left, right, valid_left, valid_right = result
    # REW's original fusion is an average in the overlap. Export the actual
    # contribution weights, separately from geometric support masks.
    mass = valid_left + valid_right
    mass[mass == 0] = 1
    weight_left, weight_right = valid_left / mass, valid_right / mass
    arrays = dict(left=left, right=right, stitched=mosaic)
    for side, weight, valid in [('left', weight_left, valid_left),
                                 ('right', weight_right, valid_right)]:
        arrays[f'mask_{side}_soft'] = np.rint(weight * 255).astype(np.uint8)
        # Match save_stage_results' > 0.5 binary convention before PNG encoding.
        arrays[f'mask_{side}_bin'] = ((weight > .5) * 255).astype(np.uint8)
        arrays[f'valid_{side}'] = (valid * 255).astype(np.uint8)
    # An averaging method has no estimated hard seam. seam_soft is its true
    # left contribution inside the overlap (0.5), zero outside the overlap.
    overlap = (valid_left > 0) & (valid_right > 0)
    arrays['seam_soft'] = np.rint(overlap * .5 * 255).astype(np.uint8)
    for name, image in arrays.items():
        write_image(files[name], image)
    files['metadata'].write_text(json.dumps({**row, **info,
        'fusion': 'upstream_REW_overlap_average',
        'seam_soft': 'constant 0.5 in overlap; no optimized seam',
        'left_x_right_x_usage': 'metadata only; REW uses SIFT correspondences',
    }, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--homography-estimator', choices=['opencv', 'legacy'], default='opencv')
    parser.add_argument('--ransac-reproj-threshold', type=float, default=3.0)
    parser.add_argument('--ransac-max-iters', type=int, default=10000)
    parser.add_argument('--min-inliers', type=int, default=6)
    parser.add_argument('--matching-mode', choices=['robust', 'legacy'], default='robust')
    parser.add_argument('--sift-ratio', type=float, default=0.7)
    parser.add_argument('--dedup-radius-px', type=float, default=1.0)
    parser.add_argument('--feature-mask-mode', choices=['content', 'none'], default='content')
    parser.add_argument('--feature-black-threshold', type=int, default=0)
    parser.add_argument('--exclude-top-px', type=int, default=0)
    parser.add_argument('--exclude-border-px', type=int, default=0)
    parser.add_argument('--geometric-validation', choices=['robust', 'legacy'], default='robust')
    parser.add_argument('--min-inlier-coverage', type=float, default=0.01)
    parser.add_argument('--max-local-scale', type=float, default=10.0)
    parser.add_argument('--allow-reflection', action='store_true')
    parser.add_argument('--max-canvas-pixels', type=int, default=8000000)
    args = parser.parse_args()
    if args.max_canvas_pixels <= 0 or not 0 <= args.seed <= 2147483647:
        parser.error('Canvas limit must be positive; seed must be in [0, 2147483647]')
    if not np.isfinite(args.ransac_reproj_threshold) or args.ransac_reproj_threshold <= 0 or args.ransac_max_iters < 1 or args.min_inliers < 4:
        parser.error('RANSAC threshold must be finite and positive, iterations >= 1, min inliers >= 4')
    if not np.isfinite(args.sift_ratio) or not 0 < args.sift_ratio < 1 or not np.isfinite(args.dedup_radius_px) or args.dedup_radius_px < 0:
        parser.error('SIFT ratio must be in (0,1); dedup radius must be finite and >= 0')
    if not 0 <= args.feature_black_threshold <= 255 or min(args.exclude_top_px, args.exclude_border_px) < 0:
        parser.error('Black threshold must be in [0,255]; feature exclusions must be >= 0')
    if not np.isfinite(args.min_inlier_coverage) or not 0 <= args.min_inlier_coverage <= 1 or not np.isfinite(args.max_local_scale) or args.max_local_scale <= 1:
        parser.error('Coverage must be in [0,1]; local scale must be finite and > 1')
    matching_options = dict(matching_mode=args.matching_mode, ratio=args.sift_ratio,
                            dedup_radius=args.dedup_radius_px, mask_mode=args.feature_mask_mode,
                            black_threshold=args.feature_black_threshold,
                            exclude_top=args.exclude_top_px, exclude_border=args.exclude_border_px)
    geometry_options = dict(geometric_validation=args.geometric_validation,
                            min_inlier_coverage=args.min_inlier_coverage,
                            max_local_scale=args.max_local_scale, allow_reflection=args.allow_reflection)
    try:
        rows = read_manifest(args.manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # The orchestration layer may create this directory beforehand. Permit it,
    # while preventing stale files for the requested pairs from being evaluated.
    for row in rows:
        diagnostic = args.out_dir / 'diagnostics' / row['stage'] / row['case'] / f'{row["prefix"]}_failure.json'
        if diagnostic.exists():
            parser.error(f'Output already exists: {diagnostic}')
        for path in result_files(args.out_dir, row).values():
            if path.exists():
                parser.error(f'Output already exists: {path}; use a new --out-dir')
    for name in ('rew_outputs.csv', 'rew_run_config.json'):
        if (args.out_dir / name).exists():
            parser.error(f'{name} already exists; use a new --out-dir')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'rew_run_config.json').write_text(json.dumps({
        'manifest': str(args.manifest.resolve()), 'seed': args.seed,
        'matching_options': matching_options, 'geometry_options': geometry_options,
        'max_canvas_pixels': args.max_canvas_pixels,
        'implementation': 'local SIFT + configurable homography RANSAC + robust TPS REW',
        'homography_estimator': args.homography_estimator,
        'ransac_reproj_threshold_px': args.ransac_reproj_threshold,
        'ransac_max_iters': args.ransac_max_iters, 'min_inliers': args.min_inliers,
    }, indent=2), encoding='utf8')
    failures = 0
    with (args.out_dir / 'rew_outputs.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*REQUIRED, *OPTIONAL, 'status', 'error', 'seconds', 'matches', 'inliers', 'homography_estimator', 'diagnostic'])
        writer.writeheader()
        for row in rows:
            start, files = time.perf_counter(), result_files(args.out_dir, row)
            report = {**row, 'status': 'ok', 'error': '', 'homography_estimator': args.homography_estimator}
            info = {}
            try:
                result, info = align_pair(row, args.seed, args.max_canvas_pixels,
                                          estimator=args.homography_estimator,
                                          reproj_threshold=args.ransac_reproj_threshold,
                                          max_iters=args.ransac_max_iters,
                                          min_inliers=args.min_inliers, diagnostics=info,
                                          matching_options=matching_options, geometry_options=geometry_options)
                save_result(files, row, result, info)
                report.update(matches=info['matches'], inliers=info['inliers'])
            except (ValueError, OSError, np.linalg.LinAlgError, cv.error) as exc:
                failures += 1
                report.update(status='failed', error=str(exc))
                for path in files.values():
                    path.unlink(missing_ok=True)
                diagnostic = args.out_dir / 'diagnostics' / row['stage'] / row['case'] / f'{row["prefix"]}_failure.json'
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                diagnostic.write_text(json.dumps({**row, **info, 'error': str(exc)}, ensure_ascii=False, indent=2), encoding='utf8')
                report.update(diagnostic=str(diagnostic))
                for key in ('matches', 'inliers'):
                    if key in info:
                        report[key] = info[key]
            report['seconds'] = time.perf_counter() - start
            writer.writerow(report)
            stream.flush()
            print(f'{row["stage"]}/{row["case"]}/{row["prefix"]}: {report["status"]} {report["error"]}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
