import numpy as np
from src.registration import warp_to_union


def test_union_keeps_negative_translation():
	x = np.ones((20, 30), np.float32)
	w = warp_to_union(x, x, np.array([[1, 0, -10], [0, 1, -5], [0, 0, 1.0]]))
	assert w.reference.shape == (25, 40)
	assert w.target_mask.sum() == 600
