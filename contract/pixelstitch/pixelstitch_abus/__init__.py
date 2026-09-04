"""ABUS 2-D pairwise inference adapter for the official PixelStitch model.

Submodules are intentionally not eagerly imported so utilities that only need
PyTorch (for example checkpoint validation) do not also require OpenCV.
"""
