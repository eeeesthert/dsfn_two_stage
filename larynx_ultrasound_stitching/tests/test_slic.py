import numpy as np
from src.features import compute_slic


def test_slic_count_reasonable():
	x = np.random.default_rng(0).random((128, 128), dtype=np.float32)
	l, b = compute_slic(
		x,
		{
			"n_segments": 100,
			"compactness": 10,
			"sigma": 0,
			"enforce_connectivity": True,
		},
	)
	assert 50 <= len(np.unique(l)) <= 150
	assert b.shape == x.shape
