"""Lane occupancy counting from overhead camera (MOG2 or static background difference)."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

from config import AppConfig


def _roi_tuple(roi: list[int]) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    return int(x), int(y), int(w), int(h)


def center_in_rect(cx: float, cy: float, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def filter_boxes_outside_intersection(
    boxes: list[tuple[int, int, int, int]],
    intersection_rect: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    """Keep boxes whose center is NOT inside intersection_rect (treat as still on approach)."""
    if intersection_rect is None:
        return boxes
    out: list[tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        if not center_in_rect(cx, cy, intersection_rect):
            out.append((x, y, w, h))
    return out


def lane_counts_from_boxes(boxes: list[tuple[int, int, int, int]], cfg: AppConfig) -> dict[str, int]:
    """Count vehicles per zone using box centers inside each ROI (first match wins)."""
    zone_names = [k for k in ("north", "south", "west", "east", "northL", "eastL", "southL", "westL") if k in cfg.rois]
    counts = {k: 0 for k in zone_names}
    rois = {n: _roi_tuple(cfg.rois[n]) for n in zone_names}
    for x, y, w, h in boxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        for lane, roi in rois.items():
            if center_in_rect(cx, cy, roi):
                counts[lane] += 1
                break
    return counts


def _centroid_xywh(b: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def _contour_plausible(c: np.ndarray, cfg: AppConfig) -> bool:
    a = float(cv2.contourArea(c))
    if not (cfg.min_contour_area <= a <= cfg.max_contour_area):
        return False
    x, y, w, h = cv2.boundingRect(c)
    if min(w, h) < cfg.min_bbox_side_px:
        return False
    box_area = float(w * h)
    if box_area < 1.0:
        return False
    extent = a / box_area
    if extent < cfg.min_extent:
        return False
    hull = cv2.convexHull(c)
    ha = float(cv2.contourArea(hull))
    if ha < 1.0:
        return False
    solidity = a / ha
    return solidity >= cfg.min_solidity


def objects_in_mask(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    cfg: AppConfig,
) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes for blobs in the ROI that pass area + shape filters (reduces false positives)."""
    x_off, y_off, rw, rh = roi
    crop = mask[y_off : y_off + rh, x_off : x_off + rw]
    if crop is None or crop.size == 0:
        return []

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, int, int]] = []
    for c in contours:
        if not _contour_plausible(c, cfg):
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((x_off + x, y_off + y, w, h))
    return out


def _match_centroid_to_prior(
    bbox: tuple[int, int, int, int],
    prev: list[tuple[int, int, int, int]],
    max_dist: float,
) -> bool:
    if not prev:
        return False
    cx, cy = _centroid_xywh(bbox)
    for pb in prev:
        px, py = _centroid_xywh(pb)
        if math.hypot(cx - px, cy - py) <= max_dist:
            return True
    return False


def dedup_boxes_by_centroid(
    boxes: list[tuple[int, int, int, int]],
    min_dist_px: float = 28.0,
) -> list[tuple[int, int, int, int]]:
    """Drop duplicate detections when lane ROIs overlap the same blob."""
    out: list[tuple[int, int, int, int]] = []
    for b in boxes:
        if not _match_centroid_to_prior(b, out, min_dist_px):
            out.append(b)
    return out


class LaneDetector:
    """
    Estimates how many vehicles are waiting in each approach ROI.

    MOG2 adapts to slow lighting drift; staticdiff uses one empty-frame background
    (more fragile but sometimes clearer on a uniform mat).
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._mode = cfg.detector.lower().strip()
        self._bg_sub: Any = None
        self._bg_gray: np.ndarray | None = None
        # Empty-board snapshot (blur gray) for MOG2+static union; also used as staticdiff bg.
        self._ref_gray: np.ndarray | None = None
        self._hist: dict[str, deque[int]] = {
            k: deque(maxlen=max(3, cfg.smooth_frames)) for k in cfg.rois
        }
        self._warmup_left = 45
        # Previous frame raw boxes (full frame) for two-frame confirmation
        self._prev_boxes: list[tuple[int, int, int, int]] | None = None

        if self._mode == "mog2":
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=cfg.mog2_history,
                varThreshold=cfg.mog2_var_threshold,
                detectShadows=cfg.mog2_detect_shadows,
            )
        elif self._mode == "staticdiff":
            self._bg_gray = None
        else:
            raise ValueError(
                f"Unknown detector mode: {cfg.detector!r} (use mog2 or staticdiff)"
            )

    def reset_confirmation(self) -> None:
        self._prev_boxes = None

    def _confirm_boxes(self, candidates: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not self.cfg.require_detection_confirmation:
            return candidates
        if self._prev_boxes is None:
            self._prev_boxes = list(candidates)
            return []
        confirmed = [
            b
            for b in candidates
            if _match_centroid_to_prior(
                b,
                self._prev_boxes,
                float(self.cfg.confirm_centroid_max_dist_px),
            )
        ]
        self._prev_boxes = list(candidates)
        return confirmed

    def set_empty_board_reference(self, frame_bgr: np.ndarray) -> None:
        """Remember empty mat (blur gray). MOG2: used with mog2_union_static_reference. staticdiff: full bg."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self._ref_gray = gray.copy()
        if self._mode == "staticdiff":
            self._bg_gray = gray.copy()

    def set_static_background(self, frame_bgr: np.ndarray) -> None:
        self.set_empty_board_reference(frame_bgr)

    def has_static_background(self) -> bool:
        return self._bg_gray is not None

    def has_empty_reference(self) -> bool:
        return self._ref_gray is not None

    def _build_mask_mog2(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._bg_sub is None:
            raise RuntimeError("MOG2 background subtractor not initialized")
        if self._warmup_left > 0:
            fg = self._bg_sub.apply(frame_bgr)
            self._warmup_left -= 1
        else:
            lr = self.cfg.mog2_learning_rate
            if lr is not None:
                fg = self._bg_sub.apply(frame_bgr, learningRate=float(lr))
            else:
                fg = self._bg_sub.apply(frame_bgr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        k = max(3, int(self.cfg.morph_kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        if self.cfg.mog2_union_static_reference and self._ref_gray is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = cv2.absdiff(self._ref_gray, gray)
            _, st = cv2.threshold(diff, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)
            st = cv2.morphologyEx(st, cv2.MORPH_OPEN, kernel)
            st = cv2.morphologyEx(st, cv2.MORPH_CLOSE, kernel)
            fg = cv2.bitwise_or(fg, st)
        return fg

    def _build_mask_static(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._bg_gray is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(self._bg_gray, gray)
        _, fg = cv2.threshold(diff, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        k = max(3, int(self.cfg.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        return fg

    def process(
        self,
        frame_bgr: np.ndarray,
        intersection_rect: tuple[int, int, int, int] | None = None,
    ) -> tuple[dict[str, int], np.ndarray, list[tuple[int, int, int, int]]]:
        if self._mode == "mog2":
            mask = self._build_mask_mog2(frame_bgr)
        else:
            mask = self._build_mask_static(frame_bgr)

        candidates: list[tuple[int, int, int, int]] = []
        for _name, roi in self.rois_items():
            boxes = objects_in_mask(mask, roi, self.cfg)
            candidates.extend(boxes)
        candidates = dedup_boxes_by_centroid(candidates)

        if intersection_rect is not None:
            candidates = filter_boxes_outside_intersection(candidates, intersection_rect)

        all_boxes = self._confirm_boxes(candidates)

        raw = {k: 0 for k in ("north", "south", "west", "east", "northL", "eastL", "southL", "westL") if k in self.cfg.rois}
        rois = {n: _roi_tuple(self.cfg.rois[n]) for n in raw}
        for b in all_boxes:
            cx, cy = _centroid_xywh(b)
            for lane, roi in rois.items():
                if center_in_rect(cx, cy, roi):
                    raw[lane] += 1
                    break

        for k, v in raw.items():
            self._hist[k].append(v)

        stable = {
            k: int(round(sum(self._hist[k]) / len(self._hist[k]))) for k in raw
        }
        return stable, mask, all_boxes

    def rois_items(self) -> list[tuple[str, tuple[int, int, int, int]]]:
        order = ("north", "south", "west", "east", "northL", "eastL", "southL", "westL")
        return [(n, _roi_tuple(self.cfg.rois[n])) for n in order if n in self.cfg.rois]

    def draw_rois(self, frame_bgr: np.ndarray) -> np.ndarray:
        return draw_rois_overlay(frame_bgr, self.cfg)


def draw_rois_overlay(frame_bgr: np.ndarray, cfg: AppConfig) -> np.ndarray:
    out = frame_bgr.copy()
    colors = {
        "north": (0, 255, 0),
        "south": (0, 255, 255),
        "west": (255, 0, 0),
        "east": (0, 0, 255),
        "northL": (0, 180, 0),
        "southL": (180, 180, 0),
        "westL": (180, 0, 180),
        "eastL": (0, 0, 180),
    }
    for name in ("north", "south", "west", "east", "northL", "eastL", "southL", "westL"):
        if name not in cfg.rois:
            continue
        x, y, w, h = _roi_tuple(cfg.rois[name])
        color = colors.get(name, (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            name,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out


def make_yolo_detector(
    cfg: AppConfig,
) -> Callable[[np.ndarray], tuple[dict[str, int], list[tuple[int, int, int, int]]]] | None:
    """Optional ultralytics YOLO: returns None if not installed."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return None

    model = YOLO("yolov8n.pt")

    def detect(frame_bgr: np.ndarray) -> tuple[dict[str, int], list[tuple[int, int, int, int]]]:
        results = model(frame_bgr, verbose=False, imgsz=640)
        counts = {k: 0 for k in ("north", "south", "west", "east")}
        boxes_xywh: list[tuple[int, int, int, int]] = []
        if not results:
            return counts, boxes_xywh

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return counts, boxes_xywh

        names = r0.names
        for b in r0.boxes:
            cls_id = int(b.cls[0])
            if isinstance(names, dict):
                label = names.get(cls_id, "")
            else:
                label = names[cls_id] if cls_id < len(names) else ""
            if label not in ("car", "truck", "bus"):
                continue
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            boxes_xywh.append((x1, y1, x2 - x1, y2 - y1))

        return counts, boxes_xywh

    return detect
