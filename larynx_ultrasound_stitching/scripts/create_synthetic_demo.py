"""Create a deterministic textured synthetic pair and ground-truth transform."""

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
p = argparse.ArgumentParser()
p.add_argument("--output", default="data/synthetic")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
h, w = 320, 480
y, x = np.indices((h, w))
base = 0.35 * np.sin(x / 9) + 0.25 * np.cos(y / 13) + 0.4 * rng.normal(0, 0.35, (h, w))
base = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 1)
base = (base - base.min()) / (base.max() - base.min())
for _ in range(45):
	c = tuple(rng.integers([20, 20], [w - 20, h - 20]))
	cv2.circle(base, c, int(rng.integers(3, 14)), float(rng.uniform(0.2, 1)), -1)
H = np.array([[1.01, -0.035, 38], [0.035, 1.01, 14], [7e-5, -5e-5, 1]], float)
target = cv2.warpPerspective(base, np.linalg.inv(H), (w, h))
cv2.imwrite(str(out / "reference.png"), np.uint8(base * 255))
cv2.imwrite(str(out / "target.png"), np.uint8(target * 255))
np.save(out / "H_target_to_reference.npy", H)
(out / "ground_truth.json").write_text(
	json.dumps({"H_target_to_reference": H.tolist()}, indent=2)
)
print(out)
