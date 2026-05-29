"""Chessboard-based pinhole camera calibration for monocular VO."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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


def _decode_image(file_bytes: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


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

    pattern_size = (inner_corners_cols, inner_corners_rows)
    obj_template = np.zeros((inner_corners_rows * inner_corners_cols, 3), np.float32)
    grid = np.mgrid[0:inner_corners_cols, 0:inner_corners_rows].T.reshape(-1, 2)
    obj_template[:, :2] = grid.astype(np.float32)
    obj_template *= float(square_size_m)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for raw in image_bytes_list:
        image = _decode_image(raw)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not found:
            continue

        refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        obj_points.append(obj_template.copy())
        img_points.append(refined)
        image_size = (gray.shape[1], gray.shape[0])

    if image_size is None or len(obj_points) < min_valid_images:
        raise ValueError(
            f"Need at least {min_valid_images} images with a detected chessboard; "
            f"got {len(obj_points)}."
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
