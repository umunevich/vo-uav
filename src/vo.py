from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.camera_calibration import intrinsics_to_k_matrix
from src.schemas.vo_config import VOConfigSchema


class VisualOdometry:
    def __init__(
        self,
        K: np.ndarray,
        *,
        distortion: list[float] | None = None,
        feature_params: dict[str, Any] | None = None,
        lk_params: dict[str, Any] | None = None,
        min_features_to_track: int = 40,
        noise_filter_distance: float = 0.15,
        keyframe_reset_distance: float = 1.5,
        absolute_scale: float = 1.0,
    ):
        self.K = K
        self.dist_coeffs = (
            np.array(distortion, dtype=np.float64)
            if distortion
            else None
        )

        self.lk_params = lk_params or dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = feature_params or dict(
            maxCorners=2000,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3,
        )

        self.min_features_to_track = min_features_to_track
        self.noise_filter_distance = noise_filter_distance
        self.keyframe_reset_distance = keyframe_reset_distance
        self.absolute_scale = absolute_scale

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        self.kf_frame = None
        self.kf_pts = None
        self.kf_R = np.eye(3)
        self.kf_t = np.zeros((3, 1))

        self.prev_frame = None
        self.prev_pts = None

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
        return cls(
            intrinsics_to_k_matrix(camera.fu, camera.fv, camera.cu, camera.cv),
            distortion=config.distortion,
            feature_params=feature_params,
            lk_params=lk_params,
            min_features_to_track=thresholds.min_features_to_track,
            noise_filter_distance=thresholds.noise_filter_distance,
            keyframe_reset_distance=thresholds.keyframe_reset_distance,
            absolute_scale=thresholds.absolute_scale,
        )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.dist_coeffs is None or not np.any(self.dist_coeffs):
            return img
        return cv2.undistort(img, self.K, self.dist_coeffs)

    def set_new_keyframe(self, img: np.ndarray) -> None:
        self._set_keyframe(self._preprocess(img))

    def _set_keyframe(self, frame: np.ndarray) -> None:
        self.kf_frame = frame.copy()
        self.prev_frame = frame.copy()

        self.kf_pts = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        self.prev_pts = self.kf_pts

        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        frame = self._preprocess(img)

        if (
            self.kf_frame is None
            or self.kf_pts is None
            or len(self.kf_pts) < self.min_features_to_track
        ):
            self._set_keyframe(frame)
            return self.cur_t

        cur_pts, st, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame,
            frame,
            self.prev_pts,
            None,
            **self.lk_params,
        )

        good_new = cur_pts[st == 1]
        good_kf = self.kf_pts[st == 1]

        if len(good_new) < self.min_features_to_track:
            self._set_keyframe(frame)
            return self.cur_t

        E, _mask = cv2.findEssentialMat(
            good_kf,
            good_new,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is not None and (E.shape == (3, 3) or E.shape[0] % 3 == 0):
            if E.shape[0] > 3:
                E = E[0:3, 0:3]

            _retval, R_rel, t_rel, _pose_mask = cv2.recoverPose(
                E,
                good_kf,
                good_new,
                self.K,
            )

            distance_moved = np.linalg.norm(t_rel)
            if distance_moved > self.noise_filter_distance:
                t_normalized = t_rel / distance_moved

                self.cur_t = self.kf_t + self.absolute_scale * self.kf_R.dot(t_normalized)
                self.cur_R = R_rel.dot(self.kf_R)

                if distance_moved > self.keyframe_reset_distance:
                    self._set_keyframe(frame)
                    return self.cur_t

        self.prev_frame = frame.copy()
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.kf_pts = good_kf.reshape(-1, 1, 2)

        return self.cur_t
