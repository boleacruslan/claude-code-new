"""Face landmarks, eye-open detection and skin masks (MediaPipe FaceMesh)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe C++ chatter

import mediapipe as mp  # noqa: E402

_FM = mp.solutions.face_mesh

# Eye-aspect-ratio landmark sets: (outer, inner, top1, bottom1, top2, bottom2)
LEFT_EYE_EAR = (33, 133, 160, 144, 158, 153)
RIGHT_EYE_EAR = (362, 263, 385, 380, 387, 373)

NOSE_TIP = 1
CHEEK_LEFT = 234
CHEEK_RIGHT = 454
CHIN = 152
FOREHEAD = 10


def _loop_from_edges(edges) -> list[int]:
    """Turn MediaPipe's unordered connection set into an ordered polygon."""
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    start = min(adj)
    loop = [start]
    prev, cur = None, start
    while len(loop) <= len(adj):
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            break
        step = nxt[0]
        if step == start:
            break
        loop.append(step)
        prev, cur = cur, step
    return loop


def _points(edges) -> list[int]:
    """All landmark indices touched by a connection set.

    Brows and lips are open paths / double rings rather than single loops,
    so the ordered walk above would miss points. Every polygon below is
    built with a convex hull, where ordering does not matter - the full
    point set is both simpler and complete.
    """
    return sorted({i for edge in edges for i in edge})


FACE_OVAL = _loop_from_edges(_FM.FACEMESH_FACE_OVAL)
LIPS = _points(_FM.FACEMESH_LIPS)
LEFT_EYE_RING = _points(_FM.FACEMESH_LEFT_EYE)
RIGHT_EYE_RING = _points(_FM.FACEMESH_RIGHT_EYE)
LEFT_BROW = _points(_FM.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW = _points(_FM.FACEMESH_RIGHT_EYEBROW)


@dataclass
class Face:
    """One detected face, in pixel coordinates of the image it was found in."""

    landmarks: np.ndarray             # (478, 2) float32
    bbox: tuple[int, int, int, int]   # x, y, w, h
    area_frac: float                  # face box area / frame area
    ear_left: float
    ear_right: float
    yaw: float                        # 0 = frontal, 1 = full profile
    sharpness: float = 0.0
    skin_luma: float = -1.0
    source_scale: float = 1.0         # resolution the landmarks were found at

    @property
    def ear(self) -> float:
        """Best of the two eyes - one eye may be hidden by head rotation."""
        return max(self.ear_left, self.ear_right)

    def eye_state(self, closed_below: float, open_above: float) -> str:
        if self.yaw > 0.62:
            return "profile"      # EAR is unreliable on a strong profile
        if self.ear < closed_below:
            return "closed"
        if self.ear < open_above:
            return "squint"
        return "open"


def _ear(pts: np.ndarray, idx) -> float:
    outer, inner, t1, b1, t2, b2 = idx
    width = np.linalg.norm(pts[outer] - pts[inner])
    if width < 1e-6:
        return 0.0
    v1 = np.linalg.norm(pts[t1] - pts[b1])
    v2 = np.linalg.norm(pts[t2] - pts[b2])
    return float((v1 + v2) / (2.0 * width))


def _yaw(pts: np.ndarray) -> float:
    """Rough head rotation from the asymmetry of nose-to-cheek distances."""
    left = np.linalg.norm(pts[NOSE_TIP] - pts[CHEEK_LEFT])
    right = np.linalg.norm(pts[NOSE_TIP] - pts[CHEEK_RIGHT])
    if left + right < 1e-6:
        return 0.0
    return float(abs(left - right) / (left + right))


def _build_face(pts: np.ndarray, frame_w: int, frame_h: int,
                source_scale: float) -> Face:
    oval = pts[FACE_OVAL]
    x0, y0 = oval.min(axis=0)
    x1, y1 = oval.max(axis=0)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return Face(
        landmarks=pts,
        bbox=(int(x0), int(y0), int(bw), int(bh)),
        area_frac=float(bw * bh / float(frame_w * frame_h)),
        ear_left=_ear(pts, LEFT_EYE_EAR),
        ear_right=_ear(pts, RIGHT_EYE_EAR),
        yaw=_yaw(pts),
        source_scale=source_scale,
    )


def _same_face(a: Face, b: Face) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    dx = (ax + aw / 2) - (bx + bw / 2)
    dy = (ay + ah / 2) - (by + bh / 2)
    return (dx * dx + dy * dy) ** 0.5 < 0.5 * max(aw, bw)


def _merge_faces(whole: list[Face], tiled: list[Face]) -> list[Face]:
    """Combine passes, keeping the detection made at the higher resolution.

    Overlapping tiles deliberately see the same face twice; the copy found in
    a tile carries landmarks resolved on an upscaled crop, so its eye
    measurements are the more trustworthy ones.
    """
    merged: list[Face] = []
    for face in sorted(whole + tiled, key=lambda f: f.source_scale, reverse=True):
        if not any(_same_face(face, kept) for kept in merged):
            merged.append(face)
    return merged


class FaceAnalyzer:
    """Thin wrapper over FaceMesh. Not thread-safe - one per process."""

    def __init__(self, max_faces: int = 8, min_confidence: float = 0.5):
        self._mesh = _FM.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
        )

    def detect(self, bgr: np.ndarray, tiles: int = 2,
               tile_upscale: float = 2.0, always_tile: bool = False) -> list[Face]:
        """Find faces, falling back to a tiled sweep for small ones.

        FaceMesh judges a face by its size *relative to the frame*, not in
        pixels, so a guest 10m back in a group shot is missed no matter how
        many megapixels the file has - upscaling the whole frame does nothing.
        Cutting the frame into overlapping tiles makes each face large
        relative to its tile, which is what actually recovers them.

        The tiled sweep costs extra passes, so it only runs when the plain
        pass came back empty or found nothing bigger than a small face -
        close-up portraits stay on the fast path.
        """
        faces = self._detect_in(bgr, frame_shape=bgr.shape)
        needs_sweep = always_tile or not faces or faces[0].area_frac < 0.02
        if tiles >= 2 and needs_sweep:
            faces = _merge_faces(faces, self._detect_tiled(bgr, tiles, tile_upscale))
        faces.sort(key=lambda f: f.area_frac, reverse=True)
        return faces

    def _detect_in(self, bgr: np.ndarray, frame_shape, offset=(0, 0),
                   scale: float = 1.0) -> list[Face]:
        """Run FaceMesh and map landmarks back into full-frame coordinates."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []

        frame_h, frame_w = frame_shape[:2]
        ox, oy = offset
        faces: list[Face] = []
        for lm in result.multi_face_landmarks:
            pts = np.array([(p.x * w / scale + ox, p.y * h / scale + oy)
                            for p in lm.landmark], dtype=np.float32)
            faces.append(_build_face(pts, frame_w, frame_h, scale))
        return faces

    def _detect_tiled(self, bgr: np.ndarray, grid: int, upscale: float) -> list[Face]:
        h, w = bgr.shape[:2]
        span = 1.0 / grid + 0.12          # overlap, so a face on a seam survives
        step = (1.0 - span) / (grid - 1) if grid > 1 else 0.0
        found: list[Face] = []
        for gy in range(grid):
            for gx in range(grid):
                x0, y0 = int(gx * step * w), int(gy * step * h)
                x1 = min(w, x0 + int(span * w))
                y1 = min(h, y0 + int(span * h))
                tile = bgr[y0:y1, x0:x1]
                if tile.shape[0] < 40 or tile.shape[1] < 40:
                    continue
                if upscale != 1.0:
                    tile = cv2.resize(tile, None, fx=upscale, fy=upscale,
                                      interpolation=cv2.INTER_CUBIC)
                found.extend(self._detect_in(tile, frame_shape=bgr.shape,
                                             offset=(x0, y0), scale=upscale))
        return found

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass


def primary_faces(faces: list[Face], min_area_frac: float = 0.004,
                  rel_to_largest: float = 0.30) -> list[Face]:
    """Faces that are plausibly the subject, not background guests.

    A wedding frame can hold 40 faces; judging closed eyes on a guest 20m
    back would reject every usable shot. We keep faces that are big enough
    in absolute terms *and* comparable to the largest face present.
    """
    if not faces:
        return []
    largest = faces[0].area_frac
    return [
        f for f in faces
        if f.area_frac >= min_area_frac and f.area_frac >= largest * rel_to_largest
    ]


def eye_region_box(face: Face, pad: float = 0.6) -> tuple[int, int, int, int]:
    """Box around both eyes - where a portrait's focus is supposed to land.

    Judging focus on the whole face box lets a sharp collar or sharp hair
    rescue a frame whose eyes are soft. Photographers cull on the eyes, so
    the metric should too.
    """
    pts = face.landmarks[LEFT_EYE_RING + RIGHT_EYE_RING + list(LEFT_EYE_EAR) +
                         list(RIGHT_EYE_EAR)]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    px, py = w * pad * 0.25, h * pad
    return (int(x0 - px), int(y0 - py), int(w + 2 * px), int(h + 2 * py))


def _poly(mask: np.ndarray, pts: np.ndarray, idx: list[int], value: int = 255) -> None:
    poly = pts[idx].astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(poly), value)


def skin_mask(shape: tuple[int, int], face: Face, bgr: Optional[np.ndarray] = None,
              feather: float = 0.06, use_color: bool = True) -> np.ndarray:
    """Soft 0..1 mask covering facial skin, with eyes, brows and lips removed."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = face.landmarks
    _poly(mask, pts, FACE_OVAL)

    # Punch out the features that must stay crisp.
    holes = np.zeros((h, w), np.uint8)
    for idx in (LEFT_EYE_RING, RIGHT_EYE_RING, LIPS, LEFT_BROW, RIGHT_BROW):
        _poly(holes, pts, idx)
    fw = max(face.bbox[2], 1)
    grow = max(3, int(fw * 0.035)) | 1
    holes = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(holes))

    # Pull the mask in from the face outline so hair and background never blur.
    inset = max(3, int(fw * 0.03)) | 1
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset, inset)))

    if use_color and bgr is not None:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        skin = ((cr >= 133) & (cr <= 183) & (cb >= 77) & (cb <= 127)).astype(np.uint8) * 255
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        mask = cv2.bitwise_and(mask, skin)

    blur = max(3, int(fw * feather)) | 1
    soft = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(soft, 0.0, 1.0)
