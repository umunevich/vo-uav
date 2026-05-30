from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np

from src.camera_calibration import intrinsics_to_k_matrix
from src.pose_smoother import PoseSmoother
from src.schemas.vo_config import VOConfigSchema

TrackingState = Literal["initializing", "ok", "degraded", "lost"]


class VisualOdometry:
    """
    Incremental monocular VO (prev→current frame), inspired by PTAM / ORB-SLAM
    tracking ideas at a lightweight scale:

    - Frame-to-frame essential-matrix pose (continuous motion)
    - Feature replenishment when the tracked set shrinks (ORB-SLAM policy lite)
    - Periodic keyframe refresh to avoid long-baseline LK failure
    - Graceful degradation instead of freezing the trajectory
    """

    def __init__(
        self,
        K: np.ndarray,
        *,
        distortion: list[float] | None = None,
        feature_params: dict[str, Any] | None = None,
        lk_params: dict[str, Any] | None = None,
        min_features_to_track: int = 30,
        keyframe_interval: int = 45,
        max_lost_frames: int = 8,
        absolute_scale: float = 1.0,
        min_essential_inliers: int = 12,
        min_inlier_ratio: float = 0.30,
        forward_backward_threshold: float = 4.0,
        min_parallax_px: float = 0.5,
        scale_ratio_min: float = 0.4,
        scale_ratio_max: float = 2.5,
        enable_smoothing: bool = True,
        smoothing_alpha: float = 0.45,
        max_step_per_frame: float = 1.2,
    ):
        self.K = K.astype(np.float64)
        self.dist_coeffs = (
            np.array(distortion, dtype=np.float64)
            if distortion
            else None
        )

        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = feature_params or dict(
            maxCorners=1200,
            qualityLevel=0.015,
            minDistance=10,
            blockSize=7,
        )

        self.min_features_to_track = min_features_to_track
        self.keyframe_interval = keyframe_interval
        self.max_lost_frames = max_lost_frames
        self.absolute_scale = absolute_scale
        self.min_essential_inliers = min_essential_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.forward_backward_threshold = forward_backward_threshold
        self.min_parallax_px = min_parallax_px
        self.scale_ratio_min = scale_ratio_min
        self.scale_ratio_max = scale_ratio_max

        self.smoother = PoseSmoother(
            alpha=smoothing_alpha,
            max_step=max_step_per_frame,
            enabled=enable_smoothing,
        )

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        self.prev_frame: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

        self.frames_since_keyframe = 0
        self.lost_frames = 0
        self.last_median_depth: float | None = None
        self.last_confidence = 0.0
        self.tracking_state: TrackingState = "initializing"

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "VisualOdometry":
        config = VOConfigSchema.model_validate(profile)
        camera = config.camera

        lk = config.lk_tracker
        lk_params = dict(
            winSize=(lk.winSize_w, lk.winSize_h),
            maxLevel=lk.maxLevel,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                lk.max_count,
                lk.epsilon,
            ),
        )

        feature_params = dict(
            maxCorners=config.feature_detector.maxCorners,
            qualityLevel=config.feature_detector.qualityLevel,
            minDistance=config.feature_detector.minDistance,
            blockSize=config.feature_detector.blockSize,
        )

        thresholds = config.vo_thresholds
        post = config.post_processing
        return cls(
            intrinsics_to_k_matrix(camera.fu, camera.fv, camera.cu, camera.cv),
            distortion=config.distortion,
            feature_params=feature_params,
            lk_params=lk_params,
            min_features_to_track=thresholds.min_features_to_track,
            keyframe_interval=post.keyframe_interval,
            max_lost_frames=post.max_lost_frames,
            absolute_scale=thresholds.absolute_scale,
            min_essential_inliers=post.min_essential_inliers,
            min_inlier_ratio=post.min_inlier_ratio,
            forward_backward_threshold=post.forward_backward_threshold,
            min_parallax_px=post.min_parallax_px,
            scale_ratio_min=post.scale_ratio_min,
            scale_ratio_max=post.scale_ratio_max,
            enable_smoothing=post.enable_smoothing,
            smoothing_alpha=post.smoothing_alpha,
            max_step_per_frame=post.max_step_per_frame,
        )

    @property
    def confidence(self) -> float:
        return self.last_confidence

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.dist_coeffs is None or not np.any(self.dist_coeffs):
            return img
        return cv2.undistort(img, self.K, self.dist_coeffs)

    def _detect_features(self, frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
        if pts is None or len(pts) == 0:
            return None
        return pts

    def _track_points(
        self,
        prev_frame: np.ndarray,
        frame: np.ndarray,
        prev_pts: np.ndarray,
        *,
        require_min: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        min_required = require_min if require_min is not None else self.min_features_to_track

        cur_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            frame,
            prev_pts,
            None,
            **self.lk_params,
        )

        fb_pts, fb_status, _fb_err = cv2.calcOpticalFlowPyrLK(
            frame,
            prev_frame,
            cur_pts,
            None,
            **self.lk_params,
        )

        status = status.reshape(-1).astype(bool)
        fb_status = fb_status.reshape(-1).astype(bool)
        fb_dist = np.linalg.norm(
            prev_pts.reshape(-1, 2) - fb_pts.reshape(-1, 2),
            axis=1,
        )
        mask = status & fb_status & (fb_dist < self.forward_backward_threshold)

        if mask.sum() < min_required:
            return None

        return prev_pts.reshape(-1, 2)[mask], cur_pts.reshape(-1, 2)[mask]

    def _replenish_features(
        self,
        frame: np.ndarray,
        tracked_pts: np.ndarray,
    ) -> np.ndarray:
        """Detect new corners in empty regions (ORB-SLAM-style generous spawning)."""
        target = max(self.min_features_to_track * 2, 80)
        if len(tracked_pts) >= target:
            return tracked_pts.reshape(-1, 1, 2)

        mask = np.full(frame.shape, 255, dtype=np.uint8)
        min_dist = int(self.feature_params.get("minDistance", 10))
        for pt in tracked_pts:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), min_dist, 0, -1)

        need = target - len(tracked_pts)
        params = dict(self.feature_params)
        params["maxCorners"] = need

        new_pts = cv2.goodFeaturesToTrack(frame, mask=mask, **params)
        if new_pts is None or len(new_pts) == 0:
            return tracked_pts.reshape(-1, 1, 2)

        merged = np.vstack([tracked_pts, new_pts.reshape(-1, 2)])
        return merged.reshape(-1, 1, 2)

    @staticmethod
    def _median_parallax(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))

    def _estimate_relative_pose(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self._median_parallax(pts_a, pts_b) < self.min_parallax_px:
            return None

        E, mask_e = cv2.findEssentialMat(
            pts_a,
            pts_b,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None or mask_e is None:
            return None

        if E.shape[0] > 3:
            E = E[:3, :3]

        mask_e = mask_e.ravel().astype(bool)
        if mask_e.sum() < self.min_essential_inliers:
            return None
        if mask_e.sum() / max(len(mask_e), 1) < self.min_inlier_ratio:
            return None

        pts_a_in = pts_a[mask_e]
        pts_b_in = pts_b[mask_e]

        inlier_count, R_rel, t_rel, mask_pose = cv2.recoverPose(
            E,
            pts_a_in,
            pts_b_in,
            self.K,
        )

        if inlier_count >= self.min_essential_inliers:
            pose_mask = mask_pose.ravel().astype(bool)
            return (
                R_rel,
                t_rel.reshape(3, 1),
                pts_a_in[pose_mask],
                pts_b_in[pose_mask],
            )

        # Cheirality often fails on translation-dominated / planar motion (typical webcam).
        return self._estimate_affine_fallback(pts_a_in, pts_b_in)

    def _estimate_affine_fallback(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        M, inliers = cv2.estimateAffinePartial2D(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if M is None:
            return None

        mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(pts_a), dtype=bool)
        if mask.sum() < self.min_essential_inliers:
            return None

        theta = float(np.arctan2(M[1, 0], M[0, 0]))
        R_rel, _ = cv2.Rodrigues(np.array([0.0, 0.0, theta], dtype=np.float64))

        dx = M[0, 2] / self.K[0, 0]
        dy = M[1, 2] / self.K[1, 1]
        t_rel = np.array([[dx], [dy], [1.0]], dtype=np.float64)
        t_rel /= max(float(np.linalg.norm(t_rel)), 1e-9)

        return R_rel, t_rel, pts_a[mask], pts_b[mask]

    def _estimate_scale(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        R_rel: np.ndarray,
        t_unit: np.ndarray,
    ) -> float:
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R_rel, t_unit])

        pts4d = cv2.triangulatePoints(P1, P2, pts_a.T, pts_b.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        z1 = pts3d[:, 2]
        z2 = (R_rel @ pts3d.T + t_unit).T[:, 2]
        valid = (z1 > 0.1) & (z2 > 0.1) & np.isfinite(z1) & np.isfinite(z2)

        if valid.sum() < 6:
            return self.absolute_scale

        median_depth = float(np.median(z1[valid]))
        if self.last_median_depth is None or not np.isfinite(self.last_median_depth):
            self.last_median_depth = median_depth
            return self.absolute_scale

        ratio = self.last_median_depth / max(median_depth, 1e-6)
        ratio = float(np.clip(ratio, self.scale_ratio_min, self.scale_ratio_max))
        self.last_median_depth = 0.85 * self.last_median_depth + 0.15 * median_depth
        return ratio * self.absolute_scale

    def _bootstrap(self, frame: np.ndarray) -> np.ndarray:
        self.prev_frame = frame.copy()
        self.prev_pts = self._detect_features(frame)
        self.frames_since_keyframe = 0
        self.lost_frames = 0
        self.last_median_depth = None
        self.tracking_state = "initializing"
        self.last_confidence = 0.0
        return self.smoother.update(self.cur_t)

    def _apply_motion(
        self,
        R_inc: np.ndarray,
        t_inc: np.ndarray,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
    ) -> None:
        motion = float(np.linalg.norm(t_inc))
        if motion < 1e-6:
            return

        t_unit = t_inc / motion
        scale = self._estimate_scale(pts_a, pts_b, R_inc, t_unit)
        t_scaled = t_unit * scale

        self.cur_t = self.cur_t + self.cur_R @ t_scaled
        self.cur_R = R_inc @ self.cur_R

        inlier_ratio = len(pts_b) / max(len(pts_a), 1)
        self.last_confidence = float(np.clip(inlier_ratio, 0.0, 1.0))
        self.tracking_state = "ok" if self.last_confidence > 0.4 else "degraded"
        self.lost_frames = 0

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        frame = self._preprocess(img)

        if self.prev_frame is None or self.prev_pts is None:
            return self._bootstrap(frame)

        tracked = self._track_points(self.prev_frame, frame, self.prev_pts)
        if tracked is None:
            self.lost_frames += 1
            self.tracking_state = "lost"
            self.last_confidence = max(0.0, self.last_confidence * 0.9)

            if self.lost_frames >= self.max_lost_frames:
                return self._bootstrap(frame)

            self.prev_frame = frame.copy()
            return self.smoother.update(self.cur_t)

        pts_prev, pts_cur = tracked

        pose = self._estimate_relative_pose(pts_prev, pts_cur)
        if pose is not None:
            R_inc, t_inc, pts_in_a, pts_in_b = pose
            self._apply_motion(R_inc, t_inc, pts_in_a, pts_in_b)
        else:
            self.lost_frames += 1
            self.tracking_state = "degraded"
            self.last_confidence = max(0.0, self.last_confidence * 0.85)

            if self.lost_frames >= self.max_lost_frames:
                return self._bootstrap(frame)

        self.prev_frame = frame.copy()
        self.prev_pts = self._replenish_features(frame, pts_cur)
        self.frames_since_keyframe += 1

        if self.frames_since_keyframe >= self.keyframe_interval:
            self.last_median_depth = None
            self.frames_since_keyframe = 0
            fresh_pts = self._detect_features(frame)
            if fresh_pts is not None:
                self.prev_pts = fresh_pts

        return self.smoother.update(self.cur_t)
