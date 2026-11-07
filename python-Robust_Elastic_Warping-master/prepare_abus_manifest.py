"""Prepare REW's pair manifest without loading images or running stitching."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

FIELDS = ('case', 'stage', 'slice', 'prefix', 'left', 'right', 'left_x', 'right_x')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def view_images(case, view):
    files = sorted(p for p in case.iterdir() if p.is_file()
                   and p.stem == view and p.suffix.lower() in EXTENSIONS)
    directory = case / view
    slices = sorted(p for p in directory.iterdir() if p.is_file()
                    and p.suffix.lower() in EXTENSIONS) if directory.is_dir() else []
    if len(files) > 1 or (files and slices):
        raise ValueError(f'Ambiguous view: {case / view}')
    return files or slices


def slice_index(paths):
    result = {}
    for path in paths:
        sid = path.stem.removeprefix('slice_')
        if sid in result:
            raise ValueError(f'Duplicate slice ID: {path}')
        result[sid] = path
    return result


def nipple_coordinates(case):
    path = case / 'nipple_x.txt'
    if not path.exists():
        # REW does not use these coordinates. Do not invent annotation values.
        return ['', '', '']
    values = [float(v.strip()) for v in path.read_text(encoding='utf-8-sig').strip().strip('[]').split(',')]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Expected three finite coordinates: {path}')
    return values


def prepare(root, stages):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    rows = []
    cases = sorted(p for p in root.glob('case*') if p.is_dir())
    for case in cases:
        coordinates = nipple_coordinates(case)
        for stage in stages:
            left, right = [view_images(case, f'input{i}') for i in stage]
            if not left or not right:
                print(f'WARNING: skip {case.name}/{stage}: missing input view', file=sys.stderr)
                continue
            a, b = slice_index(left), slice_index(right)
            common = sorted(a.keys() & b.keys())
            if common:
                pairs = [(sid, a[sid], b[sid]) for sid in common]
                if len(common) != len(left) or len(common) != len(right):
                    print(f'WARNING: {case.name}/{stage}: unmatched slices omitted', file=sys.stderr)
            else:
                if len(left) != len(right):
                    raise ValueError(f'{case.name}/{stage}: different counts and no common slice IDs')
                single = len(left) == 1 and left[0].parent == case and right[0].parent == case
                if not single:
                    print(f'WARNING: {case.name}/{stage}: pairing sorted filenames by position', file=sys.stderr)
                pairs = [('0000' if single else p.stem.removeprefix('slice_'), p, q)
                         for p, q in zip(left, right)]
            for sid, p, q in pairs:
                rows.append(dict(case=case.name, stage=stage, slice=sid,
                                 prefix=f'{stage}_{sid}', left=str(p), right=str(q),
                                 left_x=coordinates[int(stage[0])-1],
                                 right_x=coordinates[int(stage[1])-1]))
    if not rows:
        raise ValueError(f'No ABUS pairs found under {root}; expected case*/inputN images or slice folders')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', choices=['12', '23'], default=['12', '23'])
    args = parser.parse_args()
    try:
        rows = prepare(args.dataset_root, list(dict.fromkeys(args.stages)))
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Avoid overwriting a manifest that may belong to another experiment.
        with args.out_csv.open('x', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Saved {len(rows)} pairs to {args.out_csv}; images were not loaded or modified.')


if __name__ == '__main__':
    main()
