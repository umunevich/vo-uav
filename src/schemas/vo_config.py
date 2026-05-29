from pydantic import BaseModel, Field
from typing import Optional

class CameraIntrinsics(BaseModel):
    fu: float = Field(..., description="Focal length X (pixels)")
    fv: float = Field(..., description="Focal length Y (pixels)")
    cu: float = Field(..., description="Principal point X (pixels)")
    cv: float = Field(..., description="Principal point Y (pixels)")


class CalibrationMetadata(BaseModel):
    source: str = "chessboard"
    inner_corners_cols: Optional[int] = None
    inner_corners_rows: Optional[int] = None
    square_size_m: Optional[float] = None
    reprojection_error: Optional[float] = None
    images_used: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class CalibrationResponse(BaseModel):
    camera: CameraIntrinsics
    distortion: list[float] = Field(default_factory=list)
    reprojection_error: float
    images_used: int
    image_width: int
    image_height: int

class FeatureDetectorParams(BaseModel):
    maxCorners: int = 2000
    qualityLevel: float = 0.01
    minDistance: int = 10
    blockSize: int = 3

class LkTrackerParams(BaseModel):
    winSize_w: int = 31
    winSize_h: int = 31
    maxLevel: int = 4
    max_count: int = 30
    epsilon: float = 0.01

class VoThresholds(BaseModel):
    min_features_to_track: int = 40
    noise_filter_distance: float = 0.15
    keyframe_reset_distance: float = 1.5
    absolute_scale: float = 1.0

class VOConfigSchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="User-friendly name of the camera profile")
    camera: CameraIntrinsics
    distortion: list[float] = Field(default_factory=list, description="OpenCV distortion coefficients (k1,k2,p1,p2[,k3])")
    calibration: Optional[CalibrationMetadata] = None
    feature_detector: FeatureDetectorParams = FeatureDetectorParams()
    lk_tracker: LkTrackerParams = LkTrackerParams()
    vo_thresholds: VoThresholds = VoThresholds()

class VOConfigShortResponse(BaseModel):
    id: str
    name: str