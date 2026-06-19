from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

Array3D = np.ndarray
AxisName = Literal["z", "y", "x"]


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VOLUME_EXTS = (".npy", ".npz", ".nrrd", ".nhdr")


def _axis_index(axis: AxisName) -> int:
    return {"z": 0, "y": 1, "x": 2}[axis]



def _nrrd_dtype(header_type: str, endian: str | None) -> np.dtype:
    key = header_type.strip().lower()
    mapping = {
        "uchar": "u1",
        "unsigned char": "u1",
        "uint8": "u1",
        "signed char": "i1",
        "int8": "i1",
        "short": "i2",
        "short int": "i2",
        "int16": "i2",
        "ushort": "u2",
        "unsigned short": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "unsigned int": "u4",
        "uint32": "u4",
        "float": "f4",
        "double": "f8",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported NRRD type: {header_type}")
    dtype = np.dtype(mapping[key])
    if dtype.itemsize > 1:
        if endian is None:
            endian = "little"
        dtype = dtype.newbyteorder("<" if endian.lower().startswith("little") else ">")
    return dtype


def _parse_nrrd_header(path: Path) -> tuple[dict[str, str], int]:
    header: dict[str, str] = {}
    with path.open("rb") as f:
        magic = f.readline().decode("ascii", errors="replace").strip()
        if not magic.startswith("NRRD"):
            raise ValueError(f"Not an NRRD file: {path}")
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError(f"NRRD header missing blank terminator: {path}")
            if line in {b"\n", b"\r\n"}:
                return header, f.tell()
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            if ":=" in text:
                key, value = text.split(":=", 1)
            elif ":" in text:
                key, value = text.split(":", 1)
            else:
                continue
            header[key.strip().lower()] = value.strip()



def _parse_nrrd_vector_list(value: str) -> list[list[float] | None]:
    out: list[list[float] | None] = []
    for token in re.findall(r"none|\([^)]*\)", value, flags=re.IGNORECASE):
        if token.lower() == "none":
            out.append(None)
            continue
        nums = [float(x) for x in token.strip("()").split(",") if x.strip()]
        out.append(nums)
    return out


def _read_nrrd_spacing_zyx(path: Path) -> np.ndarray | None:
    if path.suffix.lower() not in {".nrrd", ".nhdr"}:
        return None
    header, _ = _parse_nrrd_header(path)
    if "spacings" in header:
        vals = [float(x) for x in header["spacings"].split()]
        if len(vals) >= 3:
            return np.asarray([vals[2], vals[1], vals[0]], dtype=np.float32)
    if "space directions" in header:
        vectors = _parse_nrrd_vector_list(header["space directions"])
        norms = []
        for vec in vectors[:3]:
            if vec is None:
                return None
            norms.append(float(np.linalg.norm(np.asarray(vec, dtype=np.float32))))
        if len(norms) == 3:
            return np.asarray([norms[2], norms[1], norms[0]], dtype=np.float32)
    return None


def _load_nrrd(path: Path) -> Array3D:
    header, data_offset = _parse_nrrd_header(path)
    dimension = int(header.get("dimension", "0"))
    if dimension != 3:
        raise ValueError(f"Only 3D NRRD volumes are supported, got dimension={dimension}: {path}")
    sizes = tuple(int(x) for x in header["sizes"].split())
    if len(sizes) != 3:
        raise ValueError(f"NRRD sizes must have 3 values: {path}")
    dtype = _nrrd_dtype(header["type"], header.get("endian"))
    encoding = header.get("encoding", "raw").strip().lower()
    data_file = header.get("data file") or header.get("datafile")

    if data_file:
        raw_path = (path.parent / data_file).resolve()
        raw = raw_path.read_bytes()
    else:
        raw = path.read_bytes()[data_offset:]

    if encoding in {"gzip", "gz"}:
        raw = gzip.decompress(raw)
        arr = np.frombuffer(raw, dtype=dtype)
    elif encoding in {"raw", ""}:
        arr = np.frombuffer(raw, dtype=dtype)
    elif encoding in {"ascii", "text", "txt"}:
        arr = np.fromstring(raw.decode("ascii"), sep=" ", dtype=dtype)
    else:
        raise ValueError(f"Unsupported NRRD encoding {encoding!r}: {path}")

    expected = int(np.prod(sizes))
    if arr.size < expected:
        raise ValueError(f"NRRD data too short: expected {expected} values, got {arr.size}: {path}")
    arr_xyz = arr[:expected].reshape(sizes, order="F")
    return np.transpose(arr_xyz, (2, 1, 0)).astype(np.float32)


def _load_image_stack(path: Path) -> Array3D:
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No image slices found in {path}")
    slices = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        slices.append(img.astype(np.float32))
    return np.stack(slices, axis=0)


def _load_volume(path: Path) -> Array3D:
    if path.is_dir():
        return _load_image_stack(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = data.files[0]
        arr = data[key].astype(np.float32)
    elif path.suffix.lower() in {".nrrd", ".nhdr"}:
        arr = _load_nrrd(path)
    else:
        raise ValueError(
            f"Unsupported volume format: {path}. Use .npy, .npz, .nrrd/.nhdr, or a directory of image slices."
        )
    if arr.ndim != 3:
        raise ValueError(f"Volume must be 3D [z,y,x], got shape={arr.shape}: {path}")
    return arr


def _find_view_path(case_dir: Path, name: str) -> Path:
    direct = case_dir / name
    if direct.exists():
        return direct
    for ext in VOLUME_EXTS:
        p = case_dir / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find view {name!r} under {case_dir}")


def _infer_nipple_y_from_xz(
    volume: Array3D,
    x: float,
    z: float,
    threshold_percentile: float,
    x_radius: int,
    z_radius: int,
) -> float:
    zi = int(round(float(z)))
    xi = int(round(float(x)))
    z0 = max(0, zi - z_radius)
    z1 = min(volume.shape[0], zi + z_radius + 1)
    x0 = max(0, xi - x_radius)
    x1 = min(volume.shape[2], xi + x_radius + 1)
    if z0 >= z1 or x0 >= x1:
        return (volume.shape[1] - 1) / 2.0

    patch = volume[z0:z1, :, x0:x1]
    profile = np.median(patch, axis=(0, 2))
    threshold = float(np.percentile(volume, threshold_percentile))
    foreground = np.flatnonzero(profile > threshold)
    if foreground.size == 0:
        return (volume.shape[1] - 1) / 2.0
    return float(foreground[0])


def _parse_landmark(
    raw: object,
    coord_order: str,
    volume: Array3D,
    nipple_y_mode: str,
    nipple_y_threshold_percentile: float,
    nipple_y_x_radius: int,
    nipple_y_z_radius: int,
) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape == (3,):
        if coord_order == "zyx":
            return arr
        if coord_order == "xyz":
            return arr[::-1]
    if arr.shape == (2,):
        zyx = np.zeros(3, dtype=np.float32)
        if coord_order == "xz":
            zyx[2] = arr[0]
            zyx[0] = arr[1]
        elif coord_order == "zx":
            zyx[0] = arr[0]
            zyx[2] = arr[1]
        else:
            zyx[0] = np.nan
        if not np.isnan(zyx[0]):
            if nipple_y_mode == "foreground-start":
                zyx[1] = _infer_nipple_y_from_xz(
                    volume,
                    x=zyx[2],
                    z=zyx[0],
                    threshold_percentile=nipple_y_threshold_percentile,
                    x_radius=nipple_y_x_radius,
                    z_radius=nipple_y_z_radius,
                )
            elif nipple_y_mode == "center":
                zyx[1] = (volume.shape[1] - 1) / 2.0
            else:
                raise ValueError(f"Unsupported nipple y mode: {nipple_y_mode}")
            return zyx
    raise ValueError(
        f"Nipple coordinate must be length 3 ({coord_order}=zyx/xyz) "
        f"or length 2 ({coord_order}=xz/zx), got {raw}"
    )


def _load_landmarks(
    case_dir: Path,
    landmark_file: str,
    nipple_root: str | None,
    coord_order: str,
    volumes: dict[str, Array3D],
    nipple_y_mode: str,
    nipple_y_threshold_percentile: float,
    nipple_y_x_radius: int,
    nipple_y_z_radius: int,
) -> dict[str, np.ndarray]:
    if nipple_root:
        path = Path(nipple_root) / f"{case_dir.name}_nipple.txt"
    else:
        path = case_dir / landmark_file
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        values = []
        for key in ("input1", "input2", "input3"):
            if key in data:
                values.append(data[key].get("nipple") if isinstance(data[key], dict) else data[key])
        if not values:
            # Backward-compatible JSON names: LAT/AP/MED map to input1/input2/input3.
            for key in ("LAT", "AP", "MED"):
                if key not in data:
                    raise KeyError(f"{path} missing {key} nipple coordinate")
                value = data[key]
                values.append(value.get("nipple") if isinstance(value, dict) else value)
    else:
        values = ast.literal_eval(text)
    if len(values) != 3:
        raise ValueError(f"{path} must contain 3 nipple coordinates for input1/input2/input3")
    return {
        "LEFT": _parse_landmark(
            values[0],
            coord_order=coord_order,
            volume=volumes["LEFT"],
            nipple_y_mode=nipple_y_mode,
            nipple_y_threshold_percentile=nipple_y_threshold_percentile,
            nipple_y_x_radius=nipple_y_x_radius,
            nipple_y_z_radius=nipple_y_z_radius,
        ),
        "CENTER": _parse_landmark(
            values[1],
            coord_order=coord_order,
            volume=volumes["CENTER"],
            nipple_y_mode=nipple_y_mode,
            nipple_y_threshold_percentile=nipple_y_threshold_percentile,
            nipple_y_x_radius=nipple_y_x_radius,
            nipple_y_z_radius=nipple_y_z_radius,
        ),
        "RIGHT": _parse_landmark(
            values[2],
            coord_order=coord_order,
            volume=volumes["RIGHT"],
            nipple_y_mode=nipple_y_mode,
            nipple_y_threshold_percentile=nipple_y_threshold_percentile,
            nipple_y_x_radius=nipple_y_x_radius,
            nipple_y_z_radius=nipple_y_z_radius,
        ),
    }


def _symmetric_spatial_weight(
    rel_phys: np.ndarray,
    weight_axis: AxisName,
    d0_mm: float,
    transition_mm: float,
    weight_mode: str,
) -> np.ndarray:
    axis = _axis_index(weight_axis)
    side_distance = np.abs(rel_phys[..., axis])
    transition = max(float(transition_mm), 1e-6)
    if weight_mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-(side_distance - float(d0_mm)) / transition))
    if weight_mode == "smoothstep":
        t = np.clip((side_distance - float(d0_mm)) / transition, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"Unsupported weight mode: {weight_mode}")


def _rotation_matrix(axis: AxisName, angle_rad: np.ndarray) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    zeros = np.zeros_like(c)
    ones = np.ones_like(c)
    rot = np.zeros(angle_rad.shape + (3, 3), dtype=np.float32)
    if axis == "z":
        rot[..., 0, 0] = c
        rot[..., 0, 1] = -s
        rot[..., 1, 0] = s
        rot[..., 1, 1] = c
        rot[..., 2, 2] = ones
    elif axis == "y":
        rot[..., 0, 0] = c
        rot[..., 0, 2] = s
        rot[..., 1, 1] = ones
        rot[..., 2, 0] = -s
        rot[..., 2, 2] = c
    elif axis == "x":
        rot[..., 0, 0] = ones
        rot[..., 1, 1] = c
        rot[..., 1, 2] = -s
        rot[..., 2, 1] = s
        rot[..., 2, 2] = c
    else:
        raise ValueError(f"Unsupported rotation axis: {axis}")
    _ = zeros  # keep c/s/zeros/ones shape checks obvious for linters
    return rot


def _spatial_weight(
    rel_phys: np.ndarray,
    weight_axis: AxisName,
    side_sign: float,
    d0_mm: float,
    transition_mm: float,
    weight_mode: str,
) -> np.ndarray:
    axis = _axis_index(weight_axis)
    side_distance = side_sign * rel_phys[..., axis]
    transition = max(float(transition_mm), 1e-6)
    if weight_mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-(side_distance - float(d0_mm)) / transition))
    if weight_mode == "smoothstep":
        t = np.clip((side_distance - float(d0_mm)) / transition, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"Unsupported weight mode: {weight_mode}")


def _trilinear_sample(volume: Array3D, coords_zyx: np.ndarray, fill_value: float = 0.0) -> Array3D:
    z = coords_zyx[..., 0]
    y = coords_zyx[..., 1]
    x = coords_zyx[..., 2]
    d, h, w = volume.shape

    z0 = np.floor(z).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    valid = (z0 >= 0) & (y0 >= 0) & (x0 >= 0) & (z1 < d) & (y1 < h) & (x1 < w)
    z0c = np.clip(z0, 0, d - 1)
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    z1c = np.clip(z1, 0, d - 1)
    y1c = np.clip(y1, 0, h - 1)
    x1c = np.clip(x1, 0, w - 1)

    dz = (z - z0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    dx = (x - x0).astype(np.float32)

    c000 = volume[z0c, y0c, x0c]
    c001 = volume[z0c, y0c, x1c]
    c010 = volume[z0c, y1c, x0c]
    c011 = volume[z0c, y1c, x1c]
    c100 = volume[z1c, y0c, x0c]
    c101 = volume[z1c, y0c, x1c]
    c110 = volume[z1c, y1c, x0c]
    c111 = volume[z1c, y1c, x1c]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    out = c0 * (1 - dz) + c1 * dz
    out = out.astype(np.float32)
    out[~valid] = fill_value
    return out


def hinge_prewarp_volume(
    src: Array3D,
    output_shape: tuple[int, int, int],
    src_nipple_zyx: np.ndarray,
    dst_nipple_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    theta_deg: float,
    rotation_axis: AxisName,
    weight_axis: AxisName,
    side_sign: float,
    d0_mm: float,
    transition_mm: float,
    weight_mode: str,
    soft_lateral_scale: float,
    soft_depth_shift_mm: float,
    soft_depth_axis: AxisName,
    fill_value: float = 0.0,
) -> Array3D:
    """
    Prewarp a source volume into the AP/reference grid with a nipple-anchored
    weighted hinge transform.

    The transform uses inverse mapping for stable resampling. The spatially
    varying rotation is approximated by computing the weight in output/reference
    coordinates, then applying the inverse local rotation to find source samples.
    """
    zz, yy, xx = np.meshgrid(
        np.arange(output_shape[0], dtype=np.float32),
        np.arange(output_shape[1], dtype=np.float32),
        np.arange(output_shape[2], dtype=np.float32),
        indexing="ij",
    )
    out_vox = np.stack([zz, yy, xx], axis=-1)
    rel_phys = (out_vox - dst_nipple_zyx.reshape(1, 1, 1, 3)) * spacing_zyx.reshape(1, 1, 1, 3)

    weights = _spatial_weight(
        rel_phys,
        weight_axis=weight_axis,
        side_sign=side_sign,
        d0_mm=d0_mm,
        transition_mm=transition_mm,
        weight_mode=weight_mode,
    ).astype(np.float32)

    # Low-DOF soft-tissue approximation.  This is intentionally conservative:
    # it smoothly scales the lateral distance and shifts depth only where the
    # hinge weight is active, avoiding free-form deformation without labels.
    rel_for_inverse = rel_phys.copy()
    lateral_axis = _axis_index(weight_axis)
    depth_axis = _axis_index(soft_depth_axis)
    if abs(float(soft_lateral_scale)) > 1e-8:
        scale = 1.0 + float(soft_lateral_scale) * weights
        scale = np.clip(scale, 0.25, 4.0)
        rel_for_inverse[..., lateral_axis] = rel_for_inverse[..., lateral_axis] / scale
    if abs(float(soft_depth_shift_mm)) > 1e-8:
        rel_for_inverse[..., depth_axis] = rel_for_inverse[..., depth_axis] - float(soft_depth_shift_mm) * weights

    angles = np.deg2rad(-float(theta_deg)) * weights
    rot = _rotation_matrix(rotation_axis, angles)
    src_rel_phys = np.einsum("...ij,...j->...i", rot, rel_for_inverse)
    src_vox = src_rel_phys / spacing_zyx.reshape(1, 1, 1, 3) + src_nipple_zyx.reshape(1, 1, 1, 3)
    return _trilinear_sample(src, src_vox, fill_value=fill_value)


def soft_deform_reference_volume(
    src: Array3D,
    nipple_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    weight_axis: AxisName,
    d0_mm: float,
    transition_mm: float,
    weight_mode: str,
    soft_lateral_scale: float,
    soft_depth_shift_mm: float,
    soft_depth_axis: AxisName,
    fill_value: float = 0.0,
) -> Array3D:
    zz, yy, xx = np.meshgrid(
        np.arange(src.shape[0], dtype=np.float32),
        np.arange(src.shape[1], dtype=np.float32),
        np.arange(src.shape[2], dtype=np.float32),
        indexing="ij",
    )
    out_vox = np.stack([zz, yy, xx], axis=-1)
    rel_phys = (out_vox - nipple_zyx.reshape(1, 1, 1, 3)) * spacing_zyx.reshape(1, 1, 1, 3)
    weights = _symmetric_spatial_weight(
        rel_phys,
        weight_axis=weight_axis,
        d0_mm=d0_mm,
        transition_mm=transition_mm,
        weight_mode=weight_mode,
    ).astype(np.float32)

    rel_for_inverse = rel_phys.copy()
    lateral_axis = _axis_index(weight_axis)
    depth_axis = _axis_index(soft_depth_axis)
    if abs(float(soft_lateral_scale)) > 1e-8:
        scale = 1.0 + float(soft_lateral_scale) * weights
        scale = np.clip(scale, 0.25, 4.0)
        rel_for_inverse[..., lateral_axis] = rel_for_inverse[..., lateral_axis] / scale
    if abs(float(soft_depth_shift_mm)) > 1e-8:
        rel_for_inverse[..., depth_axis] = rel_for_inverse[..., depth_axis] - float(soft_depth_shift_mm) * weights

    src_vox = rel_for_inverse / spacing_zyx.reshape(1, 1, 1, 3) + nipple_zyx.reshape(1, 1, 1, 3)
    return _trilinear_sample(src, src_vox, fill_value=fill_value)


def _normalize_to_u8(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo + 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - lo) / (hi - lo)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _slice_with_slab(volume: Array3D, axis: AxisName, idx: int, slab: int, mode: str) -> np.ndarray:
    ax = _axis_index(axis)
    half = slab // 2
    start = max(0, idx - half)
    end = min(volume.shape[ax], idx + half + 1)
    block = np.take(volume, indices=np.arange(start, end), axis=ax)
    if mode == "mean":
        return block.mean(axis=ax)
    if mode == "max":
        return block.max(axis=ax)
    if mode == "median":
        return np.median(block, axis=ax).astype(np.float32)
    raise ValueError(f"Unsupported slab mode: {mode}")


def export_slices(
    volume: Array3D,
    out_dir: Path,
    axis: AxisName,
    slab: int,
    slab_mode: str,
    p_low: float,
    p_high: float,
    start_index: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = volume.shape[_axis_index(axis)]
    start = min(max(int(start_index), 0), max(n - 1, 0))
    for out_i, src_i in enumerate(range(start, n)):
        img = _slice_with_slab(volume, axis=axis, idx=src_i, slab=slab, mode=slab_mode)
        img_u8 = _normalize_to_u8(img, p_low=p_low, p_high=p_high)
        cv2.imwrite(str(out_dir / f"slice_{out_i:04d}.png"), img_u8)


def projected_nipple_x(nipple_zyx: np.ndarray, export_axis: AxisName) -> float:
    # The exported image keeps the remaining axes in numpy order.
    # export_axis="y" produces coronal xz images, whose horizontal axis is volume x.
    # export_axis="z" produces transverse xy images, whose horizontal axis is also volume x.
    # export_axis="x" produces zy images, whose horizontal axis is volume y.
    if export_axis in {"z", "y"}:
        return float(nipple_zyx[2])
    return float(nipple_zyx[1])


def _crop_from_z(volume: Array3D, start_z: int) -> Array3D:
    if start_z < 0 or start_z >= volume.shape[0]:
        raise ValueError(f"start_z={start_z} out of range for volume Z={volume.shape[0]}")
    return volume[start_z:, :, :]


def _truncate_to_same_z(volumes: dict[str, Array3D]) -> dict[str, Array3D]:
    min_z = min(v.shape[0] for v in volumes.values())
    return {k: v[:min_z, :, :] for k, v in volumes.items()}


def _align_z_by_nipple_crop(
    volumes: dict[str, Array3D],
    landmarks: dict[str, np.ndarray],
) -> tuple[dict[str, Array3D], dict[str, np.ndarray]]:
    nipple_z = {k: int(round(float(v[0]))) for k, v in landmarks.items()}
    target_z = min(nipple_z.values())
    cropped: dict[str, Array3D] = {}
    updated: dict[str, np.ndarray] = {}
    for key, vol in volumes.items():
        start_z = nipple_z[key] - target_z
        cropped[key] = _crop_from_z(vol, start_z)
        updated[key] = landmarks[key].copy()
        updated[key][0] = updated[key][0] - start_z
    cropped = _truncate_to_same_z(cropped)
    return cropped, updated


def process_case(args: argparse.Namespace, case_dir: Path) -> None:
    left_path = _find_view_path(case_dir, args.left_name)
    center_path = _find_view_path(case_dir, args.center_name)
    right_path = _find_view_path(case_dir, args.right_name)
    left = _load_volume(left_path)
    center = _load_volume(center_path)
    right = _load_volume(right_path)
    if left.ndim != 3 or center.ndim != 3 or right.ndim != 3:
        raise ValueError(f"All views must be 3D volumes: {case_dir}")

    volumes = {"LEFT": left, "CENTER": center, "RIGHT": right}
    landmarks = _load_landmarks(
        case_dir,
        args.landmark_file,
        args.nipple_root,
        coord_order=args.coord_order,
        volumes=volumes,
        nipple_y_mode=args.nipple_y_mode,
        nipple_y_threshold_percentile=args.nipple_y_threshold_percentile,
        nipple_y_x_radius=args.nipple_y_x_radius,
        nipple_y_z_radius=args.nipple_y_z_radius,
    )
    if args.align_z_by_nipple_crop:
        volumes, landmarks = _align_z_by_nipple_crop(volumes, landmarks)
        left = volumes["LEFT"]
        center = volumes["CENTER"]
        right = volumes["RIGHT"]

    if args.spacing is None:
        spacing = _read_nrrd_spacing_zyx(center_path)
        if spacing is None:
            spacing = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        spacing = np.asarray(args.spacing, dtype=np.float32)
    fill_value = float(np.percentile(center, args.fill_percentile))

    left_pre = hinge_prewarp_volume(
        left,
        output_shape=center.shape,
        src_nipple_zyx=landmarks["LEFT"],
        dst_nipple_zyx=landmarks["CENTER"],
        spacing_zyx=spacing,
        theta_deg=args.left_theta,
        rotation_axis=args.rotation_axis,
        weight_axis=args.weight_axis,
        side_sign=args.left_side_sign,
        d0_mm=args.d0_mm,
        transition_mm=args.transition_mm,
        weight_mode=args.weight_mode,
        soft_lateral_scale=args.soft_lateral_scale,
        soft_depth_shift_mm=args.soft_depth_shift_mm,
        soft_depth_axis=args.soft_depth_axis,
        fill_value=fill_value,
    )
    right_pre = hinge_prewarp_volume(
        right,
        output_shape=center.shape,
        src_nipple_zyx=landmarks["RIGHT"],
        dst_nipple_zyx=landmarks["CENTER"],
        spacing_zyx=spacing,
        theta_deg=args.right_theta,
        rotation_axis=args.rotation_axis,
        weight_axis=args.weight_axis,
        side_sign=args.right_side_sign,
        d0_mm=args.d0_mm,
        transition_mm=args.transition_mm,
        weight_mode=args.weight_mode,
        soft_lateral_scale=args.soft_lateral_scale,
        soft_depth_shift_mm=args.soft_depth_shift_mm,
        soft_depth_axis=args.soft_depth_axis,
        fill_value=fill_value,
    )

    center_pre = center
    if abs(float(args.center_soft_lateral_scale)) > 1e-8 or abs(float(args.center_soft_depth_shift_mm)) > 1e-8:
        center_pre = soft_deform_reference_volume(
            center,
            nipple_zyx=landmarks["CENTER"],
            spacing_zyx=spacing,
            weight_axis=args.weight_axis,
            d0_mm=args.d0_mm,
            transition_mm=args.transition_mm,
            weight_mode=args.weight_mode,
            soft_lateral_scale=args.center_soft_lateral_scale,
            soft_depth_shift_mm=args.center_soft_depth_shift_mm,
            soft_depth_axis=args.soft_depth_axis,
            fill_value=fill_value,
        )

    start_index = 0
    if args.start_at_nipple_z and args.export_axis == "z":
        start_index = int(round(float(landmarks["CENTER"][0])))

    out_case = Path(args.out_root) / case_dir.name
    export_slices(
        left_pre, out_case / "input1", args.export_axis, args.slab, args.slab_mode, args.p_low, args.p_high, start_index
    )
    export_slices(
        center_pre, out_case / "input2", args.export_axis, args.slab, args.slab_mode, args.p_low, args.p_high, start_index
    )
    export_slices(
        right_pre, out_case / "input3", args.export_axis, args.slab, args.slab_mode, args.p_low, args.p_high, start_index
    )

    nipple_x = projected_nipple_x(landmarks["CENTER"], args.export_axis)
    (out_case / "nipple_x.txt").write_text(
        f"[{nipple_x:.3f}, {nipple_x:.3f}, {nipple_x:.3f}]\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Preprocess raw 3D ABUS input1/input2/input3 volumes with nipple-anchored "
            "weighted hinge transforms and export 2D slices for the existing pairwise pipeline."
        )
    )
    ap.add_argument("--input-root", required=True, help="root containing case*/ subdirectories")
    ap.add_argument("--out-root", required=True, help="output dataset root compatible with ABUSPairDataset")
    ap.add_argument("--left-name", default="input1", help="left view filename stem or directory name")
    ap.add_argument("--center-name", default="input2", help="center/AP view filename stem or directory name")
    ap.add_argument("--right-name", default="input3", help="right view filename stem or directory name")
    ap.add_argument("--landmark-file", default="nipple.txt")
    ap.add_argument(
        "--nipple-root",
        default=None,
        help="optional directory containing <case_name>_nipple.txt files, e.g. case_mapping/nipple_coordinates",
    )
    ap.add_argument("--coord-order", choices=["xz", "zx", "zyx", "xyz"], default="xz")
    ap.add_argument(
        "--nipple-y-mode",
        choices=["foreground-start", "center"],
        default="foreground-start",
        help="infer missing y from the transverse xy plane at each x/z nipple point or use volume center",
    )
    ap.add_argument(
        "--nipple-y-threshold-percentile",
        type=float,
        default=5.0,
        help="foreground threshold percentile used when --nipple-y-mode foreground-start",
    )
    ap.add_argument("--nipple-y-x-radius", type=int, default=2)
    ap.add_argument("--nipple-y-z-radius", type=int, default=1)
    ap.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("SZ", "SY", "SX"),
        help="voxel spacing in internal z/y/x order; defaults to center input2 NRRD spacing when available",
    )
    ap.add_argument("--left-theta", type=float, default=-15.0, help="left/input1 hinge angle in degrees")
    ap.add_argument("--right-theta", type=float, default=15.0, help="right/input3 hinge angle in degrees")
    ap.add_argument("--rotation-axis", choices=["z", "y", "x"], default="y")
    ap.add_argument("--weight-axis", choices=["z", "y", "x"], default="x")
    ap.add_argument("--left-side-sign", type=float, default=-1.0)
    ap.add_argument("--right-side-sign", type=float, default=1.0)
    ap.add_argument("--d0-mm", type=float, default=20.0, help="nipple-protection distance before hinge rotation starts")
    ap.add_argument("--transition-mm", type=float, default=10.0, help="transition width for weighted rotation")
    ap.add_argument(
        "--weight-mode",
        choices=["smoothstep", "sigmoid"],
        default="smoothstep",
        help="smoothstep keeps the nipple-side protected zone at zero rotation; sigmoid is the previous soft weighting",
    )
    ap.add_argument(
        "--soft-lateral-scale",
        type=float,
        default=0.0,
        help="optional smooth lateral compression/stretch in hinge-active tissue; 0 disables",
    )
    ap.add_argument(
        "--soft-depth-shift-mm",
        type=float,
        default=0.0,
        help="optional smooth depth shift in millimeters in hinge-active tissue; 0 disables",
    )
    ap.add_argument(
        "--center-soft-lateral-scale",
        type=float,
        default=0.0,
        help="optional symmetric lateral compression/stretch for center/input2; 0 keeps input2 unchanged",
    )
    ap.add_argument(
        "--center-soft-depth-shift-mm",
        type=float,
        default=0.0,
        help="optional symmetric depth shift for center/input2; 0 keeps input2 unchanged",
    )
    ap.add_argument("--soft-depth-axis", choices=["z", "y", "x"], default="y")
    ap.add_argument(
        "--export-axis",
        choices=["z", "y", "x"],
        default="z",
        help="axis to slice along; z exports transverse xy images, y exports coronal xz images",
    )
    ap.add_argument(
        "--align-z-by-nipple-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="crop each input from z_i - min(z1,z2,z3), then truncate to common Z length",
    )
    ap.add_argument(
        "--start-at-nipple-z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when exporting xy slices (--export-axis z), additionally make slice_0000 start at input2 nipple z",
    )
    ap.add_argument("--slab", type=int, default=1, help="odd slab size for slice projection")
    ap.add_argument("--slab-mode", choices=["mean", "max", "median"], default="mean")
    ap.add_argument("--p-low", type=float, default=1.0, help="low percentile for PNG normalization")
    ap.add_argument("--p-high", type=float, default=99.0, help="high percentile for PNG normalization")
    ap.add_argument("--fill-percentile", type=float, default=1.0, help="center/input2 percentile used for outside-volume fill")
    args = ap.parse_args()

    if args.slab < 1 or args.slab % 2 == 0:
        raise ValueError("--slab must be a positive odd integer")

    input_root = Path(args.input_root)
    case_dirs = sorted(p for p in input_root.glob("case*") if p.is_dir())
    if not case_dirs:
        raise RuntimeError(f"No case* directories found under {input_root}")
    for case_dir in case_dirs:
        process_case(args, case_dir)
        print(f"[preprocess] wrote {Path(args.out_root) / case_dir.name}")


if __name__ == "__main__":
    main()
