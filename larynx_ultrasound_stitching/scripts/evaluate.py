"""Evaluate two already aligned arrays/images."""

import argparse, json, sys, cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.image import normalize_image
from src.output import evaluate_pair

p = argparse.ArgumentParser()
p.add_argument("reference")
p.add_argument("target")
p.add_argument("--output", default="metrics.json")
a = p.parse_args()
r = normalize_image(cv2.imread(a.reference, 0))
t = normalize_image(cv2.imread(a.target, 0))
print(json.dumps(evaluate_pair(r, t, r, None), indent=2))
