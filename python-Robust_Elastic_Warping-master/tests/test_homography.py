"""Numerical and logging regressions; no ABUS stitching experiments."""
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comp_KR as estimator
import run_abus_pairwise as entry
from homography_geometry import projected_corners


def test_canvas_is_invariant_to_matrix_scale_and_sign():
    h = np.array([[1., .01, 20], [0, 1, -10], [1e-5, 0, 1]])
    for inverse in [False, True]:
        expected = projected_corners(h, (240, 320), inverse=inverse)
        for factor in [1e-20, -1e-20, 1e20, -1e20]:
            np.testing.assert_allclose(projected_corners(h*factor, (240,320), inverse=inverse), expected, atol=1e-10)


def test_actual_horizon_crossing_is_still_rejected():
    h = np.array([[1., 0, 0], [0, 1, 0], [-.01, 0, 1]])
    for factor in [1e-20, -1e20, 1]:
        with pytest.raises(ValueError, match='horizon crosses image'):
            projected_corners(h*factor, (240,320))


def test_singular_inverse_is_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        projected_corners(np.diag([1., 1, 0]), (32,32), inverse=True)


def test_pixel_ransac_recovers_transform_with_outliers():
    rng = np.random.default_rng(3)
    source = rng.uniform(0, 500, (100,2))
    target = source + [35, -12] + rng.normal(0,.1,(100,2))
    target[60:] = rng.uniform(0, 500, (40,2))
    x1 = np.vstack([source.T, np.ones(100)])
    x2 = np.vstack([target.T, np.ones(100)])
    info = {}
    cv.setRNGSeed(0)
    h, in1, _ = estimator.comp_KR(np.zeros((600,600),np.uint8),np.zeros((600,600),np.uint8),x1,x2,display=False,diagnostics=info)
    normalized = h / h[2,2]
    np.testing.assert_allclose(normalized[:2,2], [35,-12], atol=.2)
    assert 55 <= in1.shape[1] <= 60
    assert info['reprojection_p95_px'] < 1
    assert info['homography_estimator'] == 'opencv'


def test_legacy_option_uses_original_estimator(monkeypatch):
    points = np.array([[0.,10,10,0,5,2], [0,0,10,10,5,8], [1]*6])
    called = []
    def original(a,b,n,t):
        called.append((n,t))
        return np.eye(3), np.ones(6,dtype=bool), 6
    monkeypatch.setattr(estimator, 'HM_ransac', original)
    estimator.comp_KR(None,None,points,points,display=False,estimator='legacy',geometric_validation='legacy')
    assert called == [(500,.1)]


def test_failed_pair_keeps_matrix_and_counts_in_diagnostic(tmp_path, monkeypatch):
    image = tmp_path/'input.png'
    image.touch()  # Existence only; image decoding is mocked and never runs.
    manifest = tmp_path/'pairs.csv'
    with manifest.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=entry.REQUIRED)
        writer.writeheader()
        writer.writerow(dict(case='case001',stage='12',slice='0001',prefix='12_0001',left=str(image),right=str(image),left_x='',right_x=''))
    def failed(*args,diagnostics,**kwargs):
        diagnostics.update(matches=30,inliers=12,H_left_to_right=np.eye(3).tolist(),processing_stage='canvas_or_tps')
        raise ValueError('Unbounded projected canvas: horizon crosses image')
    monkeypatch.setattr(entry,'align_pair',failed)
    out=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['run_abus_pairwise.py','--manifest',str(manifest),'--out-dir',str(out)])
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 1
    data=json.loads((out/'diagnostics/12/case001/12_0001_failure.json').read_text())
    assert data['matches']==30 and data['inliers']==12
    assert data['H_left_to_right']==np.eye(3).tolist()
    assert not list(out.rglob('*_stitched.png'))
    with (out/'rew_outputs.csv').open() as stream:
        row=next(csv.DictReader(stream))
    assert row['inliers']=='12' and row['diagnostic']
