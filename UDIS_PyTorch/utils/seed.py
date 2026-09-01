"""Reproducibility controls."""
import random,numpy as np,torch
def set_seed(seed=2020): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
