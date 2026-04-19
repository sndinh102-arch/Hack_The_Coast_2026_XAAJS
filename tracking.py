"""Centroid tracker: sticky while the car is still in-frame; drop as soon as it leaves the frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _bbox_overlaps_image(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
    """True if the axis-aligned box shares any area with the image rectangle [0, w) x [0, h)."""
    x, y, w, h = bbox
    return x < frame_w and x + w > 0 and y < frame_h and y + h > 0


@dataclass
class _Track:
    bbox: tuple[int, int, int, int]
    missed: int = 0
    hits: int = 0


class CentroidTracker:
    """
    Match detections by nearest centroid. While the last bbox still overlaps the camera image,
    keep the track (even with no detections for a long time). As soon as the bbox no longer
    overlaps the frame — i.e. the car has left the view — remove that track immediately.
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_missed_inside_view: int = 1_000_000,
        min_hits_to_be_sticky: int = 8,
        max_missed_before_sticky: int = 15,
    ) -> None:
        self.max_distance = max_distance
        self.max_missed_inside_view = max_missed_inside_view
        self.min_hits_to_be_sticky = min_hits_to_be_sticky
        self.max_missed_before_sticky = max_missed_before_sticky
        self.next_id = 0
        self.tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self.next_id = 0
        self.tracks.clear()

    def _prune_left_frame(self, frame_h: int, frame_w: int) -> None:
        """Stop tracking anything that no longer intersects the image."""
        for oid in list(self.tracks.keys()):
            t = self.tracks[oid]
            if not _bbox_overlaps_image(t.bbox, frame_w, frame_h):
                del self.tracks[oid]

    def remove_ids(self, ids: set[int]) -> None:
        """Force-remove specific track IDs (e.g. if they are outside ROIs)."""
        for oid in ids:
            self.tracks.pop(oid, None)

    def _maybe_remove_stale(self, oid: int) -> None:
        """Safety cap for tracks that stay on-screen forever without detections."""
        t = self.tracks.get(oid)
        if t is None:
            return
        # If the track never got enough consistent detections, don't let it linger.
        if t.hits < self.min_hits_to_be_sticky and t.missed >= self.max_missed_before_sticky:
            del self.tracks[oid]
            return
        if t.missed >= self.max_missed_inside_view:
            del self.tracks[oid]

    def _all_outputs(self) -> list[tuple[int, int, int, int, int]]:
        return [(oid, *t.bbox) for oid, t in sorted(self.tracks.items(), key=lambda kv: kv[0])]

    def update(
        self,
        rects: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, ...],
    ) -> list[tuple[int, int, int, int, int]]:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])

        if len(rects) == 0:
            for oid in list(self.tracks.keys()):
                self.tracks[oid].missed += 1
                self._maybe_remove_stale(oid)
            self._prune_left_frame(frame_h, frame_w)
            return self._all_outputs()

        input_centroids = np.array(
            [[x + w / 2.0, y + h / 2.0] for x, y, w, h in rects],
            dtype=np.float64,
        )

        if len(self.tracks) == 0:
            for r in rects:
                oid = self.next_id
                self.next_id += 1
                x, y, w, h = r
                self.tracks[oid] = _Track(bbox=(x, y, w, h), missed=0, hits=1)
            self._prune_left_frame(frame_h, frame_w)
            return self._all_outputs()

        object_ids = list(self.tracks.keys())
        object_centroids = np.array(
            [_centroid(self.tracks[oid].bbox) for oid in object_ids],
            dtype=np.float64,
        )

        dist = np.sqrt(
            ((object_centroids[:, None, :] - input_centroids[None, :, :]) ** 2).sum(axis=2)
        )

        pairs: list[tuple[float, int, int]] = []
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                pairs.append((float(dist[i, j]), i, j))
        pairs.sort(key=lambda t: t[0])

        used_row: set[int] = set()
        used_col: set[int] = set()

        for d, row, col in pairs:
            if d > self.max_distance:
                continue
            if row in used_row or col in used_col:
                continue
            oid = object_ids[row]
            x, y, w, h = rects[col]
            self.tracks[oid].bbox = (x, y, w, h)
            self.tracks[oid].missed = 0
            self.tracks[oid].hits += 1
            used_row.add(row)
            used_col.add(col)

        for row in set(range(len(object_ids))) - used_row:
            oid = object_ids[row]
            self.tracks[oid].missed += 1
            self._maybe_remove_stale(oid)

        for col in set(range(len(rects))) - used_col:
            oid = self.next_id
            self.next_id += 1
            x, y, w, h = rects[col]
            self.tracks[oid] = _Track(bbox=(x, y, w, h), missed=0, hits=1)

        self._prune_left_frame(frame_h, frame_w)
        return self._all_outputs()
