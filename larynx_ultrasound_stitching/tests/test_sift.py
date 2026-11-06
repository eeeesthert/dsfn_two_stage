import cv2, numpy as np
from src.features import extract_global_sift


def test_sift_detects_texture():
	x = np.zeros((128, 128), np.float32)
	cv2.circle(x, (64, 64), 25, 1, -1)
	fs = extract_global_sift(
		x,
		{
			"nfeatures": 0,
			"contrast_threshold": 0.01,
			"edge_threshold": 10,
			"sigma": 1.6,
		},
	)
	assert fs.descriptors.shape[1] == 128
	assert len(fs.features) > 0
