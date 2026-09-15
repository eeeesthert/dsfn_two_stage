import numpy as np
from src.constraints import build_gabor_filter_bank, apply_gabor_bank


def test_oriented_stripes_have_selective_response():
	y, x = np.indices((128, 128))
	im = (np.sin(x * 0.5) > 0).astype(np.float32)
	cfg = {
		"scales": 5,
		"orientations": 6,
		"sigma": 2.0,
		"kernel_size": 31,
		"backend": "paper",
	}
	r = apply_gabor_bank(im, build_gabor_filter_bank(cfg))
	e = np.mean(r * r, axis=(1, 2)).reshape(5, 6)
	assert e.max() > e.min() * 2
