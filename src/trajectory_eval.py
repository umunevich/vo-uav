"""Trajectory export (TUM) and EuRoC ground-truth evaluation."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# sequence_id -> path relative to EUROC_DATASETS_PATH
EUROC_SEQUENCES: dict[str, dict[str, str]] = {
    "V1_01_easy": {
        "label": "EuRoC V1_01_easy (Vicon room)",
        "root": "vicon_room1/V1_01_easy",
    },
    "V1_02_medium": {
        "label": "EuRoC V1_02_medium (Vicon room)",
        "root": "vicon_room1/V1_02_medium",
    },
    "V1_03_difficult": {
        "label": "EuRoC V1_03_difficult (Vicon room)",
        "root": "vicon_room1/V1_03_difficult",
    },
}

GT_REL = "mav0/state_groundtruth_estimate0/data.csv"
CAM_TS_REL = "mav0/cam0/data.csv"


@dataclass(frozen=True)
class EuRoCSequencePaths:
    sequence_id: str
    label: str
    root: Path
    ground_truth_csv: Path
    cam_timestamp_csv: Path


def datasets_root() -> Path:
    env = os.environ.get("EUROC_DATASETS_PATH")
    if env:
        return Path(env).expanduser().resolve()
    # Local dev default: repo/Datasets (parent of vo-uav)
    return (Path(__file__).resolve().parents[2] / "Datasets").resolve()


def resolve_euroc_sequence(sequence_id: str) -> EuRoCSequencePaths:
    meta = EUROC_SEQUENCES.get(sequence_id)
    if meta is None:
        raise KeyError(f"Unknown EuRoC sequence: {sequence_id}")

    root = datasets_root() / meta["root"]
    gt = root / GT_REL
    cam = root / CAM_TS_REL
    if not gt.is_file():
        raise FileNotFoundError(f"Ground truth not found: {gt}")
    if not cam.is_file():
        raise FileNotFoundError(f"Camera timestamps not found: {cam}")

    return EuRoCSequencePaths(
        sequence_id=sequence_id,
        label=meta["label"],
        root=root,
        ground_truth_csv=gt,
        cam_timestamp_csv=cam,
    )


def list_euroc_sequences() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sequence_id, meta in EUROC_SEQUENCES.items():
        root = datasets_root() / meta["root"]
        gt = root / GT_REL
        cam = root / CAM_TS_REL
        available = gt.is_file() and cam.is_file()
        frame_count = 0
        if cam.is_file():
            frame_count = len(load_cam_timestamps(cam))
        items.append(
            {
                "id": sequence_id,
                "label": meta["label"],
                "available": available,
                "frame_count": frame_count,
            }
        )
    return items


def map_frame_indices(frame_indices: list[int], n_cam: int) -> list[int]:
    """Map VO frame indices into EuRoC cam timestamp range [0, n_cam - 1]."""
    if n_cam <= 0:
        raise ValueError("EuRoC camera timestamp list is empty")
    if not frame_indices:
        return []

    max_allowed = n_cam - 1
    max_idx = max(frame_indices)
    if max_idx <= max_allowed:
        return [min(max(0, idx), max_allowed) for idx in frame_indices]

    # Video may have more VO samples than EuRoC cam frames — resample proportionally.
    scale = max_allowed / max_idx if max_idx > 0 else 0.0
    return [int(round(idx * scale)) for idx in frame_indices]


def load_ground_truth(gt_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    positions: list[list[float]] = []
    with gt_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamps.append(int(row[0]))
            positions.append([float(row[1]), float(row[2]), float(row[3])])
    return np.asarray(timestamps, dtype=np.int64), np.asarray(positions, dtype=np.float64)


def load_cam_timestamps(cam_csv: Path) -> np.ndarray:
    timestamps: list[int] = []
    with cam_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamps.append(int(row[0]))
    return np.asarray(timestamps, dtype=np.int64)


def timestamps_for_frames(
    *,
    frame_indices: list[int],
    sequence: EuRoCSequencePaths | None = None,
    fps: float = 20.0,
) -> np.ndarray:
    """Return timestamps in seconds for each VO frame index."""
    if sequence is not None:
        cam_ts = load_cam_timestamps(sequence.cam_timestamp_csv)
        mapped = map_frame_indices(frame_indices, len(cam_ts))
        out = np.zeros(len(mapped), dtype=np.float64)
        for i, idx in enumerate(mapped):
            out[i] = cam_ts[idx] / 1e9
        return out

    if fps <= 0:
        raise ValueError("fps must be positive when no EuRoC sequence is selected")
    return np.asarray(frame_indices, dtype=np.float64) / fps


def umeyama_alignment(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) alignment: target ≈ scale * (source @ R.T) + t."""
    if source.shape != target.shape:
        raise ValueError("Source and target must have the same shape")
    n = source.shape[0]
    if n < 3:
        raise ValueError("At least 3 trajectory samples are required for alignment")

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s_centered = source - mu_s
    t_centered = target - mu_t
    cov = (t_centered.T @ s_centered) / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    rot = u @ s_mat @ vt
    var_s = np.mean(np.sum(s_centered**2, axis=1))
    scale = np.trace(np.diag(d) @ s_mat) / var_s if var_s > 1e-12 else 1.0
    trans = mu_t - scale * (rot @ mu_s)
    return float(scale), rot, trans


def apply_sim3(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (scale * (points @ rotation.T)) + translation


def interpolate_ground_truth(
    gt_ts: np.ndarray,
    gt_pos: np.ndarray,
    query_ts_ns: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(query_ts_ns), 3), dtype=np.float64)
    x = gt_ts.astype(np.float64)
    q = query_ts_ns.astype(np.float64)
    for axis in range(3):
        aligned[:, axis] = np.interp(q, x, gt_pos[:, axis])
    return aligned


def positions_from_samples(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[s["x"], s["y"], s["z"]] for s in samples],
        dtype=np.float64,
    )


def frame_indices_from_samples(samples: list[dict[str, Any]]) -> list[int]:
    return [int(s["frame_index"]) for s in samples]


def write_tum_trajectory(
    timestamps_s: np.ndarray,
    positions: np.ndarray,
) -> str:
    """TUM format: timestamp tx ty tz qx qy qz qw (identity orientation)."""
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for ts, pos in zip(timestamps_s, positions, strict=True):
        lines.append(
            f"{ts:.9f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
            "0.000000 0.000000 0.000000 1.000000"
        )
    return "\n".join(lines) + "\n"


def evaluate_against_euroc(
    samples: list[dict[str, Any]],
    sequence_id: str,
) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("At least 3 trajectory samples are required for evaluation")

    sequence = resolve_euroc_sequence(sequence_id)
    est = positions_from_samples(samples)
    frame_indices = frame_indices_from_samples(samples)

    cam_ts = load_cam_timestamps(sequence.cam_timestamp_csv)
    mapped_indices = map_frame_indices(frame_indices, len(cam_ts))
    query_ts = cam_ts[mapped_indices]
    gt_ts, gt_pos = load_ground_truth(sequence.ground_truth_csv)
    gt = interpolate_ground_truth(gt_ts, gt_pos, query_ts)

    scale, rotation, translation = umeyama_alignment(est, gt)
    aligned = apply_sim3(est, scale, rotation, translation)
    errors = np.linalg.norm(aligned - gt, axis=1)

    gt_path_len = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    est_path_len = float(np.sum(np.linalg.norm(np.diff(est, axis=0), axis=1)))
    aligned_path_len = float(np.sum(np.linalg.norm(np.diff(aligned, axis=0), axis=1)))

    timestamps_s = query_ts.astype(np.float64) / 1e9
    raw_tum = write_tum_trajectory(timestamps_s, est)
    scaled_tum = write_tum_trajectory(timestamps_s, aligned)

    return {
        "sequence_id": sequence_id,
        "sequence_label": sequence.label,
        "n_samples": len(samples),
        "global_scale": scale,
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_mean_m": float(np.mean(errors)),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(np.max(errors)),
        "gt_path_length_m": gt_path_len,
        "est_path_length_raw": est_path_len,
        "est_path_length_scaled_m": aligned_path_len,
        "end_error_m": float(errors[-1]),
        "drift_pct": float(errors[-1] / gt_path_len * 100) if gt_path_len > 0 else 0.0,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "scaled_positions": aligned.tolist(),
        "ground_truth_positions": gt.tolist(),
        "timestamps_s": timestamps_s.tolist(),
        "tum_raw": raw_tum,
        "tum_scaled": scaled_tum,
    }


def export_tum(
    samples: list[dict[str, Any]],
    *,
    sequence_id: str | None = None,
    fps: float = 20.0,
    apply_global_scale: bool = False,
) -> dict[str, str]:
    if not samples:
        raise ValueError("Trajectory is empty")

    positions = positions_from_samples(samples)
    frame_indices = frame_indices_from_samples(samples)
    sequence = resolve_euroc_sequence(sequence_id) if sequence_id else None
    timestamps_s = timestamps_for_frames(
        frame_indices=frame_indices,
        sequence=sequence,
        fps=fps,
    )

    if apply_global_scale:
        if not sequence_id:
            raise ValueError("sequence_id is required when apply_global_scale is true")
        metrics = evaluate_against_euroc(samples, sequence_id)
        positions = np.asarray(metrics["scaled_positions"], dtype=np.float64)
        content = metrics["tum_scaled"]
    else:
        content = write_tum_trajectory(timestamps_s, positions)

    filename = f"trajectory_{sequence_id or 'vo'}.txt"
    return {
        "filename": filename,
        "content": content,
        "format": "tum",
    }
