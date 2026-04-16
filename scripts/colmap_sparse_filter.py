"""Filter COLMAP sparse points against SAM2 object masks.

This module reads a COLMAP binary sparse model, removes 3D points whose
tracked observations do not land inside the object mask in enough views, and
writes a filtered binary model that MILo can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scripts.config_defaults import (
    COLMAP_FILTER_ENABLED,
    COLMAP_FILTER_MASK_DILATE,
    COLMAP_FILTER_MIN_INSIDE_RATIO,
    COLMAP_FILTER_MIN_INSIDE_VIEWS,
    COLMAP_FILTER_MIN_POINTS,
)

_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}

_UINT64 = struct.Struct("<Q")
_INT32 = struct.Struct("<i")
_TRACK_ELEM = struct.Struct("<ii")
_POINT2D_ELEM = struct.Struct("<ddq")
_CAMERA_HEADER = struct.Struct("<iiQQ")
_IMAGE_HEADER = struct.Struct("<i7di")
_POINT3D_HEADER = struct.Struct("<Q3d3Bd")


@dataclass(frozen=True)
class SparseFilterSettings:
    enabled: bool = COLMAP_FILTER_ENABLED
    min_inside_views: int = COLMAP_FILTER_MIN_INSIDE_VIEWS
    min_inside_ratio: float = COLMAP_FILTER_MIN_INSIDE_RATIO
    mask_dilate: int = COLMAP_FILTER_MASK_DILATE
    min_points: int = COLMAP_FILTER_MIN_POINTS


@dataclass(frozen=True)
class SparseFilterResult:
    selected_recon_dir: Path
    filtered_recon_dir: Path | None
    used_filtered: bool
    total_points: int
    kept_points: int
    matched_images: int
    warning: str | None = None


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass(frozen=True)
class ColmapPoint3D:
    point3D_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3D: dict[int, ColmapPoint3D]


def filter_colmap_sparse_model(
    recon_dir: str | Path,
    mask_dir: str | Path,
    filtered_root: str | Path,
    settings: SparseFilterSettings,
) -> SparseFilterResult:
    """Filter a COLMAP sparse model using SAM2 masks.

    Returns a result object describing whether a filtered model was produced,
    or whether MILo should fall back to the original reconstruction.
    """
    recon_path = Path(recon_dir)
    mask_path = Path(mask_dir)
    filtered_root_path = Path(filtered_root)

    model = read_colmap_model(recon_path)
    total_points = len(model.points3D)
    matched_masks = _count_matched_masks(model.images, mask_path)

    if matched_masks < max(1, settings.min_inside_views):
        _cleanup_filtered_root(filtered_root_path)
        return SparseFilterResult(
            selected_recon_dir=recon_path,
            filtered_recon_dir=None,
            used_filtered=False,
            total_points=total_points,
            kept_points=total_points,
            matched_images=matched_masks,
            warning=(
                "matched mask count is too low for sparse filtering: "
                f"{matched_masks} < {settings.min_inside_views}"
            ),
        )

    keep_ids = _select_point_ids(model, mask_path, settings)
    kept_points = len(keep_ids)
    if kept_points < max(1, settings.min_points):
        _cleanup_filtered_root(filtered_root_path)
        return SparseFilterResult(
            selected_recon_dir=recon_path,
            filtered_recon_dir=None,
            used_filtered=False,
            total_points=total_points,
            kept_points=kept_points,
            matched_images=matched_masks,
            warning=(
                "filtered sparse model retained too few points: "
                f"{kept_points} < {settings.min_points}"
            ),
        )

    filtered_model = _build_filtered_model(model, keep_ids)
    filtered_recon_dir = filtered_root_path / "0"
    tmp_dir = filtered_root_path / "_tmp_0"
    _cleanup_path(tmp_dir)
    _cleanup_path(filtered_recon_dir)
    filtered_root_path.mkdir(parents=True, exist_ok=True)

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        write_colmap_model(filtered_model, tmp_dir)
        tmp_dir.rename(filtered_recon_dir)
    except Exception:
        _cleanup_path(tmp_dir)
        _cleanup_path(filtered_recon_dir)
        raise

    return SparseFilterResult(
        selected_recon_dir=filtered_recon_dir,
        filtered_recon_dir=filtered_recon_dir,
        used_filtered=True,
        total_points=total_points,
        kept_points=kept_points,
        matched_images=matched_masks,
    )


def read_colmap_model(recon_dir: str | Path) -> ColmapModel:
    recon_path = Path(recon_dir)
    return ColmapModel(
        cameras=_read_cameras_binary(recon_path / "cameras.bin"),
        images=_read_images_binary(recon_path / "images.bin"),
        points3D=_read_points3d_binary(recon_path / "points3D.bin"),
    )


def write_colmap_model(model: ColmapModel, recon_dir: str | Path) -> None:
    recon_path = Path(recon_dir)
    recon_path.mkdir(parents=True, exist_ok=True)
    _write_cameras_binary(recon_path / "cameras.bin", model.cameras)
    _write_images_binary(recon_path / "images.bin", model.images)
    _write_points3d_binary(recon_path / "points3D.bin", model.points3D)


def _count_matched_masks(images: dict[int, ColmapImage], mask_dir: Path) -> int:
    mask_map = _build_mask_map(mask_dir)
    return sum(
        1
        for image in images.values()
        if _resolve_mask_path(image.name, mask_map) is not None
    )


def classify_points_by_mask(
    model: ColmapModel,
    mask_dir: str | Path,
    *,
    min_inside_views: int = 2,
    min_inside_ratio: float = 0.6,
    mask_dilate: int = 0,
) -> set[int]:
    """Return 3D point IDs whose tracked 2D observations land inside masks.

    Generic multi-view voting: for each COLMAP 3D point, check how many of
    its tracked 2D observations fall inside the masks.  A point is selected
    when it has at least *min_inside_views* inside votes and the inside
    fraction exceeds *min_inside_ratio*.
    """
    mask_map = _build_mask_map(Path(mask_dir))
    mask_cache: dict[Path, np.ndarray] = {}
    keep_ids: set[int] = set()

    for point_id, point in model.points3D.items():
        inside = 0
        outside = 0
        for image_id, point2d_idx in point.track:
            image = model.images.get(image_id)
            if image is None:
                continue
            if point2d_idx < 0 or point2d_idx >= len(image.point3D_ids):
                continue

            resolved_mask = _resolve_mask_path(image.name, mask_map)
            if resolved_mask is None:
                continue
            mask = mask_cache.get(resolved_mask)
            if mask is None:
                mask = _load_mask_array(resolved_mask, mask_dilate)
                mask_cache[resolved_mask] = mask

            # Use COLMAP's tracked 2D observation directly because SAM2 masks
            # live in the original frame space, not in a reprojected one.
            x, y = image.xys[point2d_idx]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            col = int(np.rint(float(x)))
            row = int(np.rint(float(y)))
            if row < 0 or row >= mask.shape[0] or col < 0 or col >= mask.shape[1]:
                continue
            if bool(mask[row, col]):
                inside += 1
            else:
                outside += 1

        total_votes = inside + outside
        if inside < min_inside_views or total_votes <= 0:
            continue
        if (inside / total_votes) < min_inside_ratio:
            continue
        keep_ids.add(point_id)

    return keep_ids


def _select_point_ids(
    model: ColmapModel,
    mask_dir: Path,
    settings: SparseFilterSettings,
) -> set[int]:
    return classify_points_by_mask(
        model,
        mask_dir,
        min_inside_views=settings.min_inside_views,
        min_inside_ratio=settings.min_inside_ratio,
        mask_dilate=settings.mask_dilate,
    )


def _build_filtered_model(model: ColmapModel, keep_ids: set[int]) -> ColmapModel:
    filtered_points = {
        point_id: point
        for point_id, point in model.points3D.items()
        if point_id in keep_ids
    }
    filtered_images: dict[int, ColmapImage] = {}
    for image_id, image in model.images.items():
        updated_ids = np.array(image.point3D_ids, copy=True)
        if updated_ids.size > 0:
            keep_mask = np.isin(updated_ids, list(keep_ids), assume_unique=False)
            updated_ids = np.where(keep_mask, updated_ids, -1).astype(np.int64, copy=False)
        filtered_images[image_id] = ColmapImage(
            image_id=image.image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=updated_ids,
        )
    return ColmapModel(
        cameras=dict(model.cameras),
        images=filtered_images,
        points3D=filtered_points,
    )


def _build_mask_map(mask_dir: Path) -> dict[str, Path]:
    return {mask_path.stem: mask_path for mask_path in sorted(mask_dir.glob("*.png"))}


def _resolve_mask_path(image_name: str, mask_map: dict[str, Path]) -> Path | None:
    stem = Path(image_name).stem
    direct = mask_map.get(stem)
    if direct is not None:
        return direct
    digits = re.findall(r"(\d+)", stem)
    if not digits:
        return None
    for candidate in reversed(digits):
        normalized = candidate.zfill(5)
        if normalized in mask_map:
            return mask_map[normalized]
        if candidate in mask_map:
            return mask_map[candidate]
        mask_key = f"mask_{candidate}"
        if mask_key in mask_map:
            return mask_map[mask_key]
        normalized_mask_key = f"mask_{candidate.zfill(4)}"
        if normalized_mask_key in mask_map:
            return mask_map[normalized_mask_key]
    return None


def _load_mask_array(mask_path: Path, dilate_iters: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    for _ in range(max(0, int(dilate_iters))):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))
    return np.array(mask, dtype=np.uint8) > 0


def _cleanup_filtered_root(filtered_root: Path) -> None:
    _cleanup_path(filtered_root / "_tmp_0")
    _cleanup_path(filtered_root / "0")


def _cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as f:
        num_cameras = _UINT64.unpack(f.read(_UINT64.size))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _CAMERA_HEADER.unpack(
                f.read(_CAMERA_HEADER.size)
            )
            num_params = _CAMERA_MODEL_NUM_PARAMS.get(model_id)
            if num_params is None:
                raise RuntimeError(f"Unsupported COLMAP camera model id: {model_id}")
            params = struct.unpack(
                "<" + "d" * num_params,
                f.read(8 * num_params),
            )
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model_id=model_id,
                width=width,
                height=height,
                params=tuple(float(v) for v in params),
            )
    return cameras


def _write_cameras_binary(path: Path, cameras: dict[int, ColmapCamera]) -> None:
    with path.open("wb") as f:
        f.write(_UINT64.pack(len(cameras)))
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            f.write(
                _CAMERA_HEADER.pack(
                    int(camera.camera_id),
                    int(camera.model_id),
                    int(camera.width),
                    int(camera.height),
                )
            )
            if camera.params:
                f.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def _read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as f:
        num_images = _UINT64.unpack(f.read(_UINT64.size))[0]
        for _ in range(num_images):
            header = _IMAGE_HEADER.unpack(f.read(_IMAGE_HEADER.size))
            image_id = int(header[0])
            qvec = tuple(float(v) for v in header[1:5])
            tvec = tuple(float(v) for v in header[5:8])
            camera_id = int(header[8])
            name = _read_null_terminated_string(f)
            num_points2d = _UINT64.unpack(f.read(_UINT64.size))[0]
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point3D_ids = np.empty((num_points2d,), dtype=np.int64)
            for idx in range(num_points2d):
                x, y, point3d_id = _POINT2D_ELEM.unpack(f.read(_POINT2D_ELEM.size))
                xys[idx, 0] = x
                xys[idx, 1] = y
                point3D_ids[idx] = point3d_id
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def _write_images_binary(path: Path, images: dict[int, ColmapImage]) -> None:
    with path.open("wb") as f:
        f.write(_UINT64.pack(len(images)))
        for image_id in sorted(images):
            image = images[image_id]
            f.write(
                _IMAGE_HEADER.pack(
                    int(image.image_id),
                    *image.qvec,
                    *image.tvec,
                    int(image.camera_id),
                )
            )
            f.write(image.name.encode("utf-8"))
            f.write(b"\x00")
            num_points = int(len(image.point3D_ids))
            f.write(_UINT64.pack(num_points))
            for idx in range(num_points):
                x, y = image.xys[idx]
                point3d_id = int(image.point3D_ids[idx])
                f.write(_POINT2D_ELEM.pack(float(x), float(y), point3d_id))


def _read_points3d_binary(path: Path) -> dict[int, ColmapPoint3D]:
    points3d: dict[int, ColmapPoint3D] = {}
    with path.open("rb") as f:
        num_points = _UINT64.unpack(f.read(_UINT64.size))[0]
        for _ in range(num_points):
            header = _POINT3D_HEADER.unpack(f.read(_POINT3D_HEADER.size))
            point3d_id = int(header[0])
            xyz = (float(header[1]), float(header[2]), float(header[3]))
            rgb = (int(header[4]), int(header[5]), int(header[6]))
            error = float(header[7])
            track_length = _UINT64.unpack(f.read(_UINT64.size))[0]
            track = []
            for _track_idx in range(track_length):
                image_id, point2d_idx = _TRACK_ELEM.unpack(f.read(_TRACK_ELEM.size))
                track.append((int(image_id), int(point2d_idx)))
            points3d[point3d_id] = ColmapPoint3D(
                point3D_id=point3d_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                track=tuple(track),
            )
    return points3d


def _write_points3d_binary(path: Path, points3d: dict[int, ColmapPoint3D]) -> None:
    with path.open("wb") as f:
        f.write(_UINT64.pack(len(points3d)))
        for point_id in sorted(points3d):
            point = points3d[point_id]
            f.write(
                _POINT3D_HEADER.pack(
                    int(point.point3D_id),
                    *point.xyz,
                    *point.rgb,
                    float(point.error),
                )
            )
            f.write(_UINT64.pack(len(point.track)))
            for image_id, point2d_idx in point.track:
                f.write(_TRACK_ELEM.pack(int(image_id), int(point2d_idx)))


def _read_null_terminated_string(stream) -> str:
    chars = bytearray()
    while True:
        value = stream.read(1)
        if value == b"\x00":
            return chars.decode("utf-8")
        if value == b"":
            raise EOFError("Unexpected EOF while reading COLMAP image name")
        chars.extend(value)
