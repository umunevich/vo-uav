from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.camera_calibration import intrinsics_to_k_matrix
from src.pose_smoother import PoseSmoother
from src.schemas.vo_config import VOConfigSchema


class VisualOdometry:
    """
    Keyframe-based monocular VO with:
    - RANSAC essential matrix + cheirality inliers
    - Forward/backward optical-flow consistency checks
    - Median triangulation depth for relative scale between frames
    - Exponential trajectory smoothing (post-processing)
    """

    def __init__(
        self,
        K: np.ndarray,
        *,
        distortion: list[float] | None = None,
        feature_params: dict[str, Any] | None = None,
        lk_params: dict[str, Any] | None = None,
        min_features_to_track: int = 40,
        noise_filter_distance: float = 0.05,
        keyframe_reset_distance: float = 1.5,
        absolute_scale: float = 1.0,
        min_essential_inliers: int = 15,
        min_inlier_ratio: float = 0.35,
        forward_backward_threshold: float = 3.0,
        keyframe_parallax_px: float = 25.0,
        min_parallax_px: float = 0.8,
        scale_ratio_min: float = 0.4,
        scale_ratio_max: float = 2.5,
        enable_smoothing: bool = True,
        smoothing_alpha: float = 0.35,
        max_step_per_frame: float = 0.8,
    ):
        self.K = K.astype(np.float64)
        self.dist_coeffs = (
            np.array(distortion, dtype=np.float64)
            if distortion
            else None
        )

        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = feature_params or dict(
            maxCorners=1500,
            qualityLevel=0.01,
            minDistance=12,
            blockSize=7,
        )

        self.min_features_to_track = min_features_to_track
        self.noise_filter_distance = noise_filter_distance
        self.keyframe_reset_distance = keyframe_reset_distance
        self.absolute_scale = absolute_scale
        self.min_essential_inliers = min_essential_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.forward_backward_threshold = forward_backward_threshold
        self.keyframe_parallax_px = keyframe_parallax_px
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

        self.kf_frame: np.ndarray | None = None
        self.kf_pts: np.ndarray | None = None
        self.kf_R = np.eye(3)
        self.kf_t = np.zeros((3, 1))

        self.prev_frame: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

        self.last_median_depth: float | None = None
        self.last_confidence = 0.0

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
            noise_filter_distance=thresholds.noise_filter_distance,
            keyframe_reset_distance=thresholds.keyframe_reset_distance,
            absolute_scale=thresholds.absolute_scale,
            min_essential_inliers=post.min_essential_inliers,
            min_inlier_ratio=post.min_inlier_ratio,
            forward_backward_threshold=post.forward_backward_threshold,
            keyframe_parallax_px=post.keyframe_parallax_px,
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

    def _detect_features(self, frame: np.ndarray) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        if pts is None or len(pts) < self.min_features_to_track:
            return None
        return pts

    def _track_points(
        self,
        prev_frame: np.ndarray,
        frame: np.ndarray,
        prev_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
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
        fb_ok = fb_dist < self.forward_backward_threshold

        mask = status & fb_status & fb_ok
        if mask.sum() < self.min_features_to_track:
            return None

        return prev_pts.reshape(-1, 2)[mask], cur_pts.reshape(-1, 2)[mask]

    @staticmethod
    def _median_parallax(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))

    def _estimate_relative_pose(
        self,
        pts_kf: np.ndarray,
        pts_cur: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self._median_parallax(pts_kf, pts_cur) < self.min_parallax_px:
            return None

        E, mask_e = cv2.findEssentialMat(
            pts_kf,
            pts_cur,
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
        inlier_ratio = mask_e.sum() / max(len(mask_e), 1)
        if mask_e.sum() < self.min_essential_inliers or inlier_ratio < self.min_inlier_ratio:
            return None

        pts_kf_in = pts_kf[mask_e]
        pts_cur_in = pts_cur[mask_e]

        inlier_count, R_rel, t_rel, mask_pose = cv2.recoverPose(
            E,
            pts_kf_in,
            pts_cur_in,
            self.K,
        )

        if inlier_count < self.min_essential_inliers:
            return None

        pose_mask = mask_pose.ravel().astype(bool)
        return (
            R_rel,
            t_rel.reshape(3, 1),
            pts_kf_in[pose_mask],
            pts_cur_in[pose_mask],
        )

    def _estimate_scale(
        self,
        pts_kf: np.ndarray,
        pts_cur: np.ndarray,
        R_rel: np.ndarray,
        t_unit: np.ndarray,
    ) -> float:
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R_rel, t_unit])

        pts4d = cv2.triangulatePoints(P1, P2, pts_kf.T, pts_cur.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        z1 = pts3d[:, 2]
        z2 = (R_rel @ pts3d.T + t_unit).T[:, 2]
        valid = (z1 > 0.1) & (z2 > 0.1) & np.isfinite(z1) & np.isfinite(z2)

        if valid.sum() < 8:
            return self.absolute_scale

        median_depth = float(np.median(z1[valid]))
        if self.last_median_depth is None or not np.isfinite(self.last_median_depth):
            self.last_median_depth = median_depth
            return self.absolute_scale

        ratio = self.last_median_depth / max(median_depth, 1e-6)
        ratio = float(np.clip(ratio, self.scale_ratio_min, self.scale_ratio_max))
        self.last_median_depth = 0.8 * self.last_median_depth + 0.2 * median_depth
        return ratio * self.absolute_scale

    def _set_keyframe(self, frame: np.ndarray) -> None:
        self.kf_frame = frame.copy()
        self.prev_frame = frame.copy()
        self.kf_pts = self._detect_features(frame)
        self.prev_pts = self.kf_pts
        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()
        self.last_median_depth = None

    def _should_promote_keyframe(self, pts_kf: np.ndarray, pts_cur: np.ndarray) -> bool:
        return self._median_parallax(pts_kf, pts_cur) >= self.keyframe_parallax_px

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        frame = self._preprocess(img)

        if self.kf_frame is None or self.kf_pts is None:
            self._set_keyframe(frame)
            self.last_confidence = 0.0
            return self.smoother.update(self.cur_t)

        # Frame-to-frame tracking keeps the point set fresh between keyframes.
        if self.prev_frame is None or self.prev_pts is None:
            self._set_keyframe(frame)
            self.last_confidence = 0.0
            return self.smoother.update(self.cur_t)

        tracked_prev = self._track_points(self.prev_frame, frame, self.prev_pts)
        if tracked_prev is None:
            self._set_keyframe(frame)
            self.last_confidence = 0.0
            return self.smoother.update(self.cur_t)

        _prev_pts, cur_pts = tracked_prev

        # Pose is estimated from the keyframe anchor to limit drift accumulation.
        kf_to_cur = self._track_points(self.kf_frame, frame, self.kf_pts)
        if kf_to_cur is None:
            self._set_keyframe(frame)
            self.last_confidence = 0.0
            return self.smoother.update(self.cur_t)

        pts_kf, pts_cur = kf_to_cur

        pose = self._estimate_relative_pose(pts_kf, pts_cur)
        if pose is None:
            self.prev_frame = frame.copy()
            self.prev_pts = cur_pts.reshape(-1, 1, 2)
            self.last_confidence = 0.1
            return self.smoother.update(self.cur_t)

        R_rel, t_rel, pts_kf_in, pts_cur_in = pose
        motion = float(np.linalg.norm(t_rel))
        if motion < self.noise_filter_distance:
            self.prev_frame = frame.copy()
            self.prev_pts = cur_pts.reshape(-1, 1, 2)
            self.last_confidence = 0.2
            return self.smoother.update(self.cur_t)

        scale = self._estimate_scale(pts_kf_in, pts_cur_in, R_rel, t_rel / motion)
        t_scaled = (t_rel / motion) * scale

        self.cur_R = R_rel @ self.kf_R
        self.cur_t = self.kf_t + self.kf_R @ t_scaled

        inlier_ratio = len(pts_kf_in) / max(len(pts_kf), 1)
        self.last_confidence = float(np.clip(inlier_ratio, 0.0, 1.0))

        if self._should_promote_keyframe(pts_kf, pts_cur) or float(np.linalg.norm(t_scaled)) > self.keyframe_reset_distance:
            self._set_keyframe(frame)
        else:
            self.prev_frame = frame.copy()
            self.prev_pts = cur_pts.reshape(-1, 1, 2)

        return self.smoother.update(self.cur_t)
