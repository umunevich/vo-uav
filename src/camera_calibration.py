"""Chessboard-based pinhole camera calibration for monocular VO."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Common inner-corner grids (OpenCV pattern_size = cols × rows).
# EuRoC cam_checkerboard uses 6×7 (see dataset calibration sequences).
COMMON_CHESSBOARD_PATTERNS: list[tuple[int, int]] = [
    (6, 7),
    (7, 6),
    (9, 6),
    (6, 9),
    (8, 6),
    (6, 8),
    (7, 5),
    (5, 7),
    (10, 7),
    (7, 10),
    (5, 6),
    (6, 5),
    (4, 5),
    (5, 4),
]


@dataclass(frozen=True)
class CalibrationResult:
    fu: float
    fv: float
    cu: float
    cv: float
    distortion: list[float]
    reprojection_error: float
    images_used: int
    image_width: int
    image_height: int
    inner_corners_cols: int
    inner_corners_rows: int
    pattern_auto_detected: bool = False


def _decode_image(file_bytes: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """Detect inner corners; prefers findChessboardCornersSB when available."""
    cols, rows = pattern_size

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ACCURACY,
        )
        if found and corners is not None:
            return True, corners.reshape(-1, 1, 2).astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if found and corners is not None:
        return True, corners

    return False, None


def _count_pattern_detections(
    decoded_images: list[np.ndarray],
    pattern_size: tuple[int, int],
) -> int:
    count = 0
    for image in decoded_images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, _ = _find_chessboard_corners(gray, pattern_size)
        if found:
            count += 1
    return count


def _resolve_pattern_size(
    decoded_images: list[np.ndarray],
    inner_corners_cols: int,
    inner_corners_rows: int,
    *,
    min_valid_images: int,
) -> tuple[int, int, bool]:
    """Use requested pattern, or auto-detect if it matches too few images."""
    # Sample for speed when folders contain hundreds of frames (e.g. EuRoC).
    sample = decoded_images
    if len(decoded_images) > 40:
        step = max(1, len(decoded_images) // 40)
        sample = decoded_images[::step][:40]

    requested = (inner_corners_cols, inner_corners_rows)
    requested_count = _count_pattern_detections(sample, requested)
    if requested_count >= min(min_valid_images, len(sample)):
        return inner_corners_cols, inner_corners_rows, False

    best_pattern = requested
    best_count = requested_count
    for cols, rows in COMMON_CHESSBOARD_PATTERNS:
        if (cols, rows) == requested:
            continue
        count = _count_pattern_detections(sample, (cols, rows))
        if count > best_count:
            best_count = count
            best_pattern = (cols, rows)

    if best_count >= min(min_valid_images, len(sample)):
        return best_pattern[0], best_pattern[1], best_pattern != requested

    return inner_corners_cols, inner_corners_rows, False


def calibrate_from_chessboard_images(
    image_bytes_list: list[bytes],
    inner_corners_cols: int,
    inner_corners_rows: int,
    square_size_m: float,
    *,
    min_valid_images: int = 3,
) -> CalibrationResult:
    """
    Estimate pinhole intrinsics from chessboard photos.

    ``inner_corners_*`` are the number of *inner* corners per row/column
    (OpenCV ``findChessboardCorners`` pattern size).
    """
    if inner_corners_cols < 3 or inner_corners_rows < 3:
        raise ValueError("Chessboard must have at least 3 inner corners along each axis.")

    if square_size_m <= 0:
        raise ValueError("Square size must be positive (meters).")

    decoded: list[np.ndarray] = []
    for raw in image_bytes_list:
        image = _decode_image(raw)
        if image is not None:
            decoded.append(image)

    if not decoded:
        raise ValueError("Could not decode any uploaded images. Use PNG or JPEG files.")

    cols, rows, auto_detected = _resolve_pattern_size(
        decoded,
        inner_corners_cols,
        inner_corners_rows,
        min_valid_images=min_valid_images,
    )
    pattern_size = (cols, rows)

    obj_template = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_template[:, :2] = grid.astype(np.float32)
    obj_template *= float(square_size_m)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image in decoded:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = _find_chessboard_corners(gray, pattern_size)
        if not found or corners is None:
            continue

        refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=subpix_criteria,
        )

        obj_points.append(obj_template.copy())
        img_points.append(refined)
        image_size = (gray.shape[1], gray.shape[0])

    if image_size is None or len(obj_points) < min_valid_images:
        raise ValueError(
            f"Need at least {min_valid_images} images with a detected chessboard; "
            f"got {len(obj_points)}. "
            f"Tried pattern {cols}×{rows} inner corners. "
            f"EuRoC cam_checkerboard uses 6×7 (not 9×6). "
            f"Upload 3+ images with the board visible from different angles."
        )

    reproj_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    dist = dist_coeffs.reshape(-1).tolist()
    return CalibrationResult(
        fu=float(camera_matrix[0, 0]),
        fv=float(camera_matrix[1, 1]),
        cu=float(camera_matrix[0, 2]),
        cv=float(camera_matrix[1, 2]),
        distortion=dist,
        reprojection_error=float(reproj_error),
        images_used=len(obj_points),
        image_width=image_size[0],
        image_height=image_size[1],
        inner_corners_cols=cols,
        inner_corners_rows=rows,
        pattern_auto_detected=auto_detected,
    )


def intrinsics_to_k_matrix(fu: float, fv: float, cu: float, cv: float) -> np.ndarray:
    return np.array(
        [
            [fu, 0.0, cu],
            [0.0, fv, cv],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def scale_intrinsics_for_resolution(
    fu: float,
    fv: float,
    cu: float,
    cv: float,
    *,
    calib_width: int,
    calib_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    """Resize pinhole intrinsics when VO frames differ from calibration resolution."""
    if calib_width <= 0 or calib_height <= 0:
        return fu, fv, cu, cv

    sx = frame_width / calib_width
    sy = frame_height / calib_height
    return fu * sx, fv * sy, cu * sx, cv * sy
