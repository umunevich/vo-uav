"""Load and persist VO camera configuration profiles."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import yaml

from src.camera_calibration import intrinsics_to_k_matrix

STORAGE_DIR = os.environ.get("VO_CONFIG_STORAGE", "storage/configs")
os.makedirs(STORAGE_DIR, exist_ok=True)

# EuRoC MAV camera (indoor hall) — sensible default when no profile is selected.
DEFAULT_CONFIG_ID = "euroc_default"
DEFAULT_INTRINSICS = {
    "fu": 458.654,
    "fv": 457.296,
    "cu": 367.215,
    "cv": 248.375,
}


def get_file_path(config_id: str) -> str:
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "", config_id)
    return os.path.join(STORAGE_DIR, f"{clean_id}.yaml")


def ensure_default_profile() -> None:
    path = get_file_path(DEFAULT_CONFIG_ID)
    if os.path.exists(path):
        return

    payload = {
        "name": "EuRoC MAV (default)",
        "camera": DEFAULT_INTRINSICS.copy(),
        "calibration": {
            "source": "dataset",
            "note": "EuRoC Machine Hall cam0; use chessboard calibration for your own camera.",
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, default_flow_style=False, allow_unicode=True)


def load_config(config_id: str) -> dict[str, Any]:
    ensure_default_profile()
    file_path = get_file_path(config_id)
    if not os.path.exists(file_path):
        file_path = get_file_path(DEFAULT_CONFIG_ID)

    with open(file_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_camera_matrix(config_id: str | None) -> np.ndarray:
    data = load_vo_profile(config_id)
    camera = data.get("camera", DEFAULT_INTRINSICS)
    return intrinsics_to_k_matrix(
        float(camera["fu"]),
        float(camera["fv"]),
        float(camera["cu"]),
        float(camera["cv"]),
    )


def load_vo_profile(config_id: str | None) -> dict[str, Any]:
    """Return the stored VO profile for the given id, or the default profile."""
    ensure_default_profile()
    requested_id = config_id or DEFAULT_CONFIG_ID
    file_path = get_file_path(requested_id)

    if not os.path.exists(file_path):
        file_path = get_file_path(DEFAULT_CONFIG_ID)

    with open(file_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
