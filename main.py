#!/usr/bin/env python3
"""
Adaptive traffic light system: camera → vehicle counts → timing → Arduino LEDs.

Examples:
  python main.py
  python main.py --config config.json --no-serial
  python main.py --calibrate
  python main.py --vision-only
  python main.py --demo
  python main.py --detector yolo   # requires: pip install ultralytics

  MOG2: press B with an empty mat once so stopped cars stay visible (optional but recommended).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import AppConfig, load_config, save_config
from controller import AdaptiveTrafficController, Phase, summarize_demand
from detection import (
    LaneDetector,
    draw_rois_overlay,
    filter_boxes_outside_intersection,
    lane_counts_from_boxes,
    make_yolo_detector,
)
from serial_out import encode_phase, open_serial
from tracking import CentroidTracker


def _draw_phase_overlay(
    frame: np.ndarray,
    phase: Phase,
    ns_wait: int,
    ew_wait: int,
    level: str,
    adaptive: bool,
) -> None:
    lines = [
        f"Phase: {phase.name}",
        f"NS wait: {ns_wait}   EW wait: {ew_wait}   ({level})",
        f"Mode: {'adaptive' if adaptive else 'fixed-time'}",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
        y += 26


def corridor_counts_for_overlay(counts: dict[str, int]) -> tuple[int, int, int, int]:
    ns_left = int(counts.get("northL", 0)) + int(counts.get("southL", 0))
    ns_straight = int(counts.get("north", 0)) + int(counts.get("south", 0))
    ew_left = int(counts.get("westL", 0)) + int(counts.get("eastL", 0))
    ew_straight = int(counts.get("west", 0)) + int(counts.get("east", 0))
    return ns_left, ns_straight, ew_left, ew_straight


def center_fraction_roi(w: int, h: int, frac: float) -> tuple[int, int, int, int]:
    rw = max(1, int(w * frac))
    rh = max(1, int(h * frac))
    x = (w - rw) // 2
    y = (h - rh) // 2
    return x, y, rw, rh


def resolve_intersection_rect(cfg: AppConfig, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Region where vehicles are treated as 'in the crossing' — boxes disappear once the center enters here."""
    if not cfg.hide_in_intersection:
        return None
    if cfg.intersection_roi is not None:
        t = tuple(int(x) for x in cfg.intersection_roi)
        return t[0], t[1], t[2], t[3]
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return center_fraction_roi(w, h, float(cfg.intersection_auto_fraction))


def draw_intersection_overlay(frame: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 1)
    cv2.putText(
        frame,
        "intersection",
        (x, max(0, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )


def draw_tracked_boxes(
    frame: np.ndarray,
    tracked: list[tuple[int, int, int, int, int]],
) -> None:
    """Draw each tracked object with a rectangle and stable ID."""
    for oid, x, y, w, h in tracked:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"#{oid}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def prune_tracks_outside_lane_rois(
    tracked: list[tuple[int, int, int, int, int]],
    cfg: AppConfig,
) -> tuple[list[tuple[int, int, int, int, int]], set[int]]:
    """Drop tracks whose centroid is not inside any lane ROI (prevents hand/edge clutter lingering)."""
    rois = [
        tuple(cfg.rois[k])
        for k in ("north", "south", "west", "east", "northL", "eastL", "southL", "westL")
        if k in cfg.rois
    ]
    kept: list[tuple[int, int, int, int, int]] = []
    remove: set[int] = set()
    for oid, x, y, w, h in tracked:
        cx, cy = x + w / 2.0, y + h / 2.0
        inside = False
        for rx, ry, rw, rh in rois:
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                inside = True
                break
        if inside:
            kept.append((oid, x, y, w, h))
        else:
            remove.add(oid)
    return kept, remove


def run_calibrate(config_path: Path) -> None:
    cfg = load_config(config_path)
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("Could not open camera", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    order = ["north", "south", "west", "east", "northL", "eastL", "southL", "westL"]
    state: dict[str, Any] = {
        "idx": 0,
        "drag": False,
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0,
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _p: Any) -> None:
        if state["idx"] >= len(order):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = True
            state["x0"], state["y0"] = x, y
            state["x1"], state["y1"] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["x1"], state["y1"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = False
            state["x1"], state["y1"] = x, y
            x0, y0 = state["x0"], state["y0"]
            x1, y1 = state["x1"], state["y1"]
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            if rw >= 20 and rh >= 20:
                name = order[state["idx"]]
                cfg.rois[name] = [rx, ry, rw, rh]
                state["idx"] += 1

    win = "Calibrate ROIs (drag each box: N, S, W, E - then Q to save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Some OpenCV/Linux builds require the window to be shown at least once before
    # attaching mouse callbacks, otherwise: NULL window handler in cvSetMouseCallback.
    ok, frame0 = cap.read()
    if ok:
        cv2.imshow(win, frame0)
        cv2.waitKey(1)
    cv2.setMouseCallback(win, on_mouse)

    print(
        "Drag rectangles in order: north, south, west, east, northL, eastL, southL, westL."
    )
    print("Press Q to save and exit, R to reset current step.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        display = frame.copy()
        colors = [
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (0, 0, 255),
            (0, 180, 0),
            (0, 0, 180),
            (180, 180, 0),
            (180, 0, 180),
        ]
        for i, name in enumerate(order):
            x, y, w, h = cfg.rois[name]
            c = colors[i]
            cv2.rectangle(display, (x, y), (x + w, y + h), c, 2)
            cv2.putText(display, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        if state["drag"]:
            x0, y0 = state["x0"], state["y0"]
            x1, y1 = state["x1"], state["y1"]
            cv2.rectangle(display, (x0, y0), (x1, y1), (200, 200, 50), 2)

        step = min(state["idx"], len(order) - 1)
        msg = f"Draw ROI for {order[step]} ({state['idx'] + 1}/{len(order)})"
        cv2.putText(display, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            state["idx"] = max(0, state["idx"] - 1)

    cap.release()
    cv2.destroyAllWindows()
    save_config(cfg, config_path)
    print(f"Saved ROIs to {config_path}")


def run_demo(config_path: Path, adaptive: bool) -> None:
    cfg = load_config(config_path)
    ctrl = AdaptiveTrafficController(cfg.timing, adaptive=adaptive)
    t0 = time.perf_counter()

    while True:
        now = time.perf_counter()
        t = now - t0
        counts = {
            "north": max(0, int(3 + 4 * math.sin(t / 5.0))),
            "south": max(0, int(2 + 2 * math.cos(t / 7.0))),
            "east": max(0, int(1 + 5 * math.sin(t / 4.0 + 1.0))),
            "west": max(0, int(1 + 3 * math.cos(t / 6.0 + 0.5))),
            "northL": max(0, int(1 + 2 * math.sin(t / 6.0 + 0.3))),
            "southL": max(0, int(1 + 2 * math.cos(t / 8.0 + 0.6))),
            "eastL": max(0, int(1 + 2 * math.sin(t / 5.5 + 1.1))),
            "westL": max(0, int(1 + 2 * math.cos(t / 7.5 + 0.9))),
        }
        phase, _ = ctrl.tick(0.05, counts)
        ns, ew, level = summarize_demand(counts)
        print(
            f"t={t:5.1f}s  {phase.name:16}  NS={ns} EW={ew} ({level})  raw={counts}",
            flush=True,
        )
        time.sleep(0.05)


def run_main(
    config_path: Path,
    no_serial: bool,
    detector_name: str | None,
    adaptive_override: bool | None,
) -> None:
    cfg = load_config(config_path)
    if detector_name:
        cfg.detector = detector_name
    if adaptive_override is not None:
        cfg.adaptive = adaptive_override

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("Could not open camera", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    yolo_fn = None
    if cfg.detector.lower() == "yolo":
        yolo_fn = make_yolo_detector(cfg)
        if yolo_fn is None:
            print("YOLO requested but ultralytics is not installed. pip install ultralytics", file=sys.stderr)
            sys.exit(1)

    det = LaneDetector(cfg) if yolo_fn is None else None
    if cfg.detector.lower() == "staticdiff":
        print("Staticdiff: press B with an EMPTY mat to capture background.")
    elif cfg.detector.lower() == "mog2":
        print(
            "MOG2: press B once with an EMPTY mat to remember the board — "
            "helps keep boxes on cars after they stop (union + slower learning).",
            flush=True,
        )

    ctrl = AdaptiveTrafficController(cfg.timing, adaptive=cfg.adaptive)
    tracker = CentroidTracker(
        max_distance=80.0,
        max_missed_inside_view=1_000_000,
        min_hits_to_be_sticky=8,
        max_missed_before_sticky=15,
    )
    ser = None
    last_phase: Phase | None = None
    if not no_serial:
        try:
            ser = open_serial(cfg.serial_port, cfg.serial_baud)
            time.sleep(0.1)
        except Exception as e:
            print(f"Serial disabled ({e}). LEDs will not update.", file=sys.stderr)
            ser = None

    win = "Adaptive Traffic"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    prev = time.perf_counter()
    exclude_rect: tuple[int, int, int, int] | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        dt = min(0.1, max(0.001, now - prev))
        prev = now

        if exclude_rect is None and cfg.hide_in_intersection:
            exclude_rect = resolve_intersection_rect(cfg, frame.shape)

        if yolo_fn is not None:
            _, boxes = yolo_fn(frame)
            boxes = filter_boxes_outside_intersection(boxes, exclude_rect)
            counts = lane_counts_from_boxes(boxes, cfg)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            display = draw_rois_overlay(frame.copy(), cfg)
        elif det is not None:
            counts, mask, boxes = det.process(
                frame,
                intersection_rect=exclude_rect if cfg.hide_in_intersection else None,
            )
            display = det.draw_rois(frame)
        else:
            print(
                "Internal error: no detector (expected MOG2/staticdiff or YOLO).",
                file=sys.stderr,
            )
            sys.exit(1)

        if exclude_rect is not None and cfg.hide_in_intersection:
            draw_intersection_overlay(display, exclude_rect)

        tracked = tracker.update(boxes, frame.shape)
        tracked, to_remove = prune_tracks_outside_lane_rois(tracked, cfg)
        if to_remove:
            tracker.remove_ids(to_remove)
        draw_tracked_boxes(display, tracked)

        phase, _lights = ctrl.tick(dt, counts)
        ns, ew, level = summarize_demand(counts)
        _draw_phase_overlay(display, phase, ns, ew, level, cfg.adaptive)

        nsL, nsS, ewL, ewS = corridor_counts_for_overlay(counts)
        cv2.putText(
            display,
            f"NS left {nsL}  NS straight {nsS}   |   EW left {ewL}  EW straight {ewS}",
            (12, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Small HUD for debugging counts
        y = display.shape[0] - 110
        cv2.putText(
            display,
            f"N {counts.get('north', 0)}  S {counts.get('south', 0)}  W {counts.get('west', 0)}  E {counts.get('east', 0)}  |  tracks: {len(tracked)}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 255),
            2,
        )

        if ser is not None and phase != last_phase:
            try:
                ser.write(encode_phase(phase))
                ser.flush()
            except Exception as e:
                print(f"Serial write failed: {e}", file=sys.stderr)
                ser = None
        last_phase = phase

        cv2.imshow(win, display)
        small = cv2.resize(mask, None, fx=0.35, fy=0.35)
        cv2.imshow("Foreground mask (debug)", small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            ctrl.reset()
            tracker.reset()
            if det is not None:
                det.reset_confirmation()
            print("Controller + tracker + detection confirmation reset", flush=True)
        if key == ord("b") and det is not None and cfg.detector.lower() in ("staticdiff", "mog2"):
            det.set_empty_board_reference(frame)
            print("Captured empty-board reference", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive traffic light controller")
    p.add_argument("--config", type=Path, default=Path("config.json"))
    p.add_argument("--calibrate", action="store_true", help="Draw ROI rectangles and save config")
    p.add_argument("--vision-only", action="store_true", help="No Arduino serial output")
    p.add_argument("--no-serial", action="store_true", help="Never open serial port")
    p.add_argument(
        "--detector",
        choices=("mog2", "staticdiff", "yolo"),
        default=None,
        help="Override detector in config",
    )
    p.add_argument("--demo", action="store_true", help="Print simulated traffic + phases (no camera)")
    p.add_argument(
        "--fixed-time",
        action="store_true",
        help="Use fixed green duration (compare against adaptive)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        run_demo(args.config, adaptive=not args.fixed_time)
        return
    if args.calibrate:
        run_calibrate(args.config)
        return

    no_serial = args.no_serial or args.vision_only
    adaptive_override = False if args.fixed_time else None
    run_main(
        args.config,
        no_serial=no_serial,
        detector_name=args.detector,
        adaptive_override=adaptive_override,
    )


if __name__ == "__main__":
    main()
