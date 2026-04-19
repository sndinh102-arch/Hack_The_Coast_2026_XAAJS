"""Load and save JSON configuration for the traffic system."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimingConfig:
    # Straight movement
    yellow_sec: float = 3.0
    straight_low_sec: float = 10.0
    straight_high_sec: float = 25.0

    # Protected left-turn movement (blue)
    left_low_sec: float = 5.0
    left_high_sec: float = 10.0
    left_blink_sec: float = 3.0
    left_off_buffer_sec: float = 1.0

    # Safety clearance between NS and EW
    all_red_clearance_sec: float = 2.0

    # Classification: >= high_threshold vehicles -> high traffic
    high_threshold: int = 2


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 115200
    detector: str = "mog2"
    mog2_history: int = 2000
    mog2_var_threshold: float = 32.0
    mog2_detect_shadows: bool = False
    # After warmup, use this learning rate for MOG2 (smaller = slower background update = stopped
    # cars stay foreground longer). null = OpenCV default (-1) per frame.
    mog2_learning_rate: float | None = 0.004
    # If True and you pressed B with an empty mat, OR absdiff vs that snapshot into the mask so
    # cars that stop moving still differ from the empty board.
    mog2_union_static_reference: bool = True
    diff_threshold: int = 30
    morph_kernel: int = 5
    min_contour_area: int = 400
    max_contour_area: int = 8000
    # Reject speckles / thin glare: solidity = contour_area / convex_hull_area
    min_solidity: float = 0.45
    # Reject sparse scribbles: extent = contour_area / bbox_area
    min_extent: float = 0.18
    min_bbox_side_px: int = 14
    # Require a blob in roughly the same place this frame and last (kills one-frame noise).
    require_detection_confirmation: bool = True
    confirm_centroid_max_dist_px: float = 45.0
    smooth_frames: int = 7
    rois: dict[str, list[int]] = field(
        default_factory=lambda: {
            "north": [250, 40, 120, 130],
            "south": [260, 310, 120, 130],
            "west": [40, 180, 170, 100],
            "east": [420, 170, 170, 100],
            "northL": [250, 40, 120, 130],
            "eastL": [420, 170, 170, 100],
            "southL": [260, 310, 120, 130],
            "westL": [40, 180, 170, 100],
        }
    )
    timing: TimingConfig = field(default_factory=TimingConfig)
    adaptive: bool = True
    # When True, drop detections whose center is inside intersection_roi (crossing). Leave False so
    # boxes stay visible while cars wait — only enable if your intersection box is tightly cropped.
    hide_in_intersection: bool = False
    # [x, y, w, h] in pixels; use null to auto-place a centered box (see intersection_auto_fraction).
    intersection_roi: list[int] | None = None
    intersection_auto_fraction: float = 0.35

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        d = dict(data)
        timing_data = d.pop("timing", {}) or {}
        timing_fields = {f.name for f in dataclasses.fields(TimingConfig)}
        timing = TimingConfig(
            **{k: v for k, v in timing_data.items() if k in timing_fields}
        )
        rois = d.pop("rois", None)
        cfg_fields = {f.name for f in dataclasses.fields(cls) if f.name not in ("timing", "rois")}
        kwargs = {k: v for k, v in d.items() if k in cfg_fields}
        cfg = cls(**kwargs)
        cfg.timing = timing
        if rois is not None:
            cfg.rois = {str(k): list(v) for k, v in rois.items()}
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "serial_port": self.serial_port,
            "serial_baud": self.serial_baud,
            "detector": self.detector,
            "mog2_history": self.mog2_history,
            "mog2_var_threshold": self.mog2_var_threshold,
            "mog2_detect_shadows": self.mog2_detect_shadows,
            "mog2_learning_rate": self.mog2_learning_rate,
            "mog2_union_static_reference": self.mog2_union_static_reference,
            "diff_threshold": self.diff_threshold,
            "morph_kernel": self.morph_kernel,
            "min_contour_area": self.min_contour_area,
            "max_contour_area": self.max_contour_area,
            "min_solidity": self.min_solidity,
            "min_extent": self.min_extent,
            "min_bbox_side_px": self.min_bbox_side_px,
            "require_detection_confirmation": self.require_detection_confirmation,
            "confirm_centroid_max_dist_px": self.confirm_centroid_max_dist_px,
            "smooth_frames": self.smooth_frames,
            "rois": self.rois,
            "timing": {
                "yellow_sec": self.timing.yellow_sec,
                "straight_low_sec": self.timing.straight_low_sec,
                "straight_high_sec": self.timing.straight_high_sec,
                "left_low_sec": self.timing.left_low_sec,
                "left_high_sec": self.timing.left_high_sec,
                "left_blink_sec": self.timing.left_blink_sec,
                "left_off_buffer_sec": self.timing.left_off_buffer_sec,
                "all_red_clearance_sec": self.timing.all_red_clearance_sec,
                "high_threshold": self.timing.high_threshold,
            },
            "adaptive": self.adaptive,
            "hide_in_intersection": self.hide_in_intersection,
            "intersection_roi": self.intersection_roi,
            "intersection_auto_fraction": self.intersection_auto_fraction,
        }


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return AppConfig.from_dict(data)


def save_config(cfg: AppConfig, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
