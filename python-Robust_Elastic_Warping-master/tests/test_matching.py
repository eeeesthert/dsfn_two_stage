"""Matching and geometry regressions on synthetic points/mocked descriptors."""
import csv
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import load_data as matching
import comp_KR as geometry
import run_abus_pairwise as entry
from homography_geometry import validate_local_geometry


def match(a,b,d=1):
    return cv.DMatch(_queryIdx=a,_trainIdx=b,_distance=d)


def test_mutual_matching_removes_many_to_one():
    forward=[match(0,0),match(1,0),match(2,2)]
    reverse=[match(0,0),match(2,2)]
    assert [(m.queryIdx,m.trainIdx) for m in matching.mutual_matches(forward,reverse)] == [(0,0),(2,2)]


def test_coordinate_dedup_handles_different_ids_and_bin_boundaries():
    source=np.array([[.9,1],[1.1,1],[4,4],[7,7],[10,10]])
    target=np.array([[0.,0],[2,2],[3,3],[3.2,3.1],[9,9]])
    assert matching.unique_correspondences(source,target,[5,1,2,4,3],radius=1.) == [1,2,4]


def test_reported_eight_inliers_have_only_four_target_locations():
    source=np.array([[222,24],[237,24],[240,24],[265,24],[271,24],[428,74],[803,3],[803,15]],float)
    target=np.array([[601,24]]*5+[[298,83],[803,3],[802,15]],float)
    assert len(matching.unique_correspondences(source,target,np.arange(8))) == 4


def test_feature_masks_do_not_modify_or_resize_images():
    gray=np.full((20,30),100,np.uint8);gray[10,10]=0
    original=gray.copy()
    external=np.full_like(gray,255);external[:,20:]=0
    mask=matching.feature_mask(gray,exclude_top=3,exclude_border=2,external=external)
    assert mask.shape==gray.shape
    assert not mask[:3].any() and not mask[:,20:].any() and mask[10,10]==0
    assert mask[5,5]==255
    np.testing.assert_array_equal(gray,original)
    with pytest.raises(ValueError,match='dimensions'):
        matching.feature_mask(gray,external=np.ones((10,10)))


def test_ratio_rejects_ambiguous_and_short_neighbor_lists():
    result=matching.ratio_matches([[match(0,0,1),match(0,1,2)],
                                   [match(1,1,2),match(1,0,2)], [match(2,2)]],.7)
    assert [m.queryIdx for m in result]==[0]


def test_failed_matching_keeps_filter_counts(monkeypatch):
    points=[cv.KeyPoint(float(5+i*5),10.,1.) for i in range(4)]
    class Sift:
        def detectAndCompute(self,*args):
            return points,np.ones((4,128),np.float32)
    class Matcher:
        calls=0
        def knnMatch(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:
                return [[match(i,0),match(i,1,10)] for i in range(4)]
            return [[match(0,0),match(0,1,10)]]
    monkeypatch.setattr(matching.cv,'SIFT_create',lambda:Sift())
    monkeypatch.setattr(matching.cv,'BFMatcher',lambda *args:Matcher())
    info={}
    with pytest.raises(ValueError,match='after robust filtering: 1'):
        matching.load_data(np.ones((40,40),np.uint8),np.ones((40,40),np.uint8),diagnostics=info)
    assert info['forward_ratio_matches']==4 and info['mutual_matches']==1 and info['matches']==1


def test_symmetric_error_rejects_compressed_false_consensus(monkeypatch):
    source=np.array([[0.,10,20,0,10,20],[0,0,0,10,10,10],[1]*6])
    h=np.diag([.01,.01,1.])
    target=h@source;target[0]+=.5
    monkeypatch.setattr(geometry.cv,'findHomography',lambda *a,**k:(h,np.ones((6,1),np.uint8)))
    info={}
    with pytest.raises(ValueError,match='Insufficient RANSAC inliers: 0'):
        geometry.comp_KR(np.zeros((30,30)),np.zeros((30,30)),source,target,display=False,diagnostics=info)
    assert info['forward_only_inliers']==6 and info['inliers']==0


def test_thin_inlier_distribution_is_rejected(monkeypatch):
    source=np.array([[5.,25,45,65,85,95],[5,5,5.1,5.1,5,5.1],[1]*6])
    monkeypatch.setattr(geometry.cv,'findHomography',lambda *a,**k:(np.eye(3),np.ones((6,1),np.uint8)))
    with pytest.raises(ValueError,match='spatial inlier coverage'):
        geometry.comp_KR(np.zeros((100,100)),np.zeros((100,100)),source,source,display=False)


def test_local_scale_and_reflection_limits_are_configurable():
    with pytest.raises(ValueError,match='compression/stretch'):
        validate_local_geometry(np.diag([.01,.01,1]),(100,100))
    reflection=np.array([[-1.,0,99],[0,1,0],[0,0,1]])
    with pytest.raises(ValueError,match='reflection'):
        validate_local_geometry(reflection,(100,100))
    assert validate_local_geometry(reflection,(100,100),allow_reflection=True)['reflected']


def test_optional_external_mask_paths_are_resolved(tmp_path):
    (tmp_path/'image.jpg').touch();(tmp_path/'mask.png').touch()
    manifest=tmp_path/'pairs.csv'
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=[*entry.REQUIRED,*entry.OPTIONAL]);writer.writeheader()
        writer.writerow(dict(case='case001',stage='12',slice='0001',prefix='12_0001',left='image.jpg',right='image.jpg',left_x='',right_x='',left_mask='mask.png',right_mask=''))
    row=entry.read_manifest(manifest)[0]
    assert row['left_mask']==str(tmp_path/'mask.png') and 'right_mask' not in row


def test_legacy_matching_preserves_one_way_flann_without_dedup(monkeypatch):
    points=[cv.KeyPoint(float(5+i*5),10.,1.) for i in range(4)]
    class Sift:
        def detectAndCompute(self,*args):
            return points,np.ones((4,128),np.float32)
    class Matcher:
        calls=0
        def knnMatch(self,*args,**kwargs):
            self.calls+=1
            assert self.calls==1
            return [[match(i,0),match(i,1,10)] for i in range(4)]
    monkeypatch.setattr(matching.cv,'SIFT_create',lambda:Sift())
    monkeypatch.setattr(matching.cv,'FlannBasedMatcher',lambda *args:Matcher())
    source,target=matching.load_data(np.ones((40,40),np.uint8),np.ones((40,40),np.uint8),matching_mode='legacy',mask_mode='none')
    assert source.shape[1]==4 and len(np.unique(target[:2].T,axis=0))==1
