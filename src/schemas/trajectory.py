from __future__ import annotations

from pydantic import BaseModel, Field


class TrajectorySample(BaseModel):
    frame_index: int = Field(ge=0)
    x: float
    y: float
    z: float
    confidence: float | None = None
    tracking: str | None = None


class EuRoCSequenceInfo(BaseModel):
    id: str
    label: str
    available: bool


class TrajectoryExportRequest(BaseModel):
    samples: list[TrajectorySample]
    sequence_id: str | None = None
    fps: float = Field(default=20.0, gt=0)
    apply_global_scale: bool = False


class TrajectoryExportResponse(BaseModel):
    filename: str
    content: str
    format: str = "tum"


class TrajectoryEvaluateRequest(BaseModel):
    samples: list[TrajectorySample]
    sequence_id: str


class TrajectoryEvaluateResponse(BaseModel):
    sequence_id: str
    sequence_label: str
    n_samples: int
    global_scale: float
    ate_rmse_m: float
    ate_mean_m: float
    ate_median_m: float
    ate_max_m: float
    gt_path_length_m: float
    est_path_length_raw: float
    est_path_length_scaled_m: float
    end_error_m: float
    drift_pct: float
    tum_raw: str
    tum_scaled: str
    scaled_positions: list[list[float]]
