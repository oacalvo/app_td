from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

MAX_PANELS = 4
CUBE_AXIS_ORDERS = {
    "TYX": (0, 1, 2),
    "TXY": (0, 2, 1),
    "YTX": (1, 0, 2),
    "YXT": (1, 2, 0),
    "XTY": (2, 0, 1),
    "XYT": (2, 1, 0),
}
CUBE_AXIS_ORDER_ALIASES = {
    "123": "TYX",
    "132": "TXY",
    "213": "YTX",
    "231": "YXT",
    "312": "XTY",
    "321": "XYT",
}

@dataclass
class Cut:
    cut_id: int
    name: str
    color: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    visible: bool = True
    locked: bool = False
    mode: str = "line"
    curve_fit: str = "line"
    curve_points: list[tuple[float, float]] = field(default_factory=list)
    function_expr: str = ""
    function_x_start: float = 0.0
    function_x_end: float = 0.0
    function_point_count: int = 0
    function_control_points: list[tuple[float, float]] = field(default_factory=list)
    function_params: dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureAxis:
    axis_id: int
    name: str
    color: str
    points: list[tuple[float, float]]
    mode: str = "curve"
    visible: bool = True


@dataclass
class TDPanel:
    panel_id: int
    name: str
    cut_id: int | None = None
    t_ini: int = 0
    t_fin: int = 0
    stride: int = 1
    width: int = 1
    weighting: str = "uniform"
    cache_key: tuple[Any, ...] | None = None
    cache_td: np.ndarray | None = field(default=None, repr=False)
    cache_meta: dict[str, Any] | None = field(default=None, repr=False)


def normalize_cube_axis_order(value: str) -> str:
    text = str(value or "").strip().upper().replace("-", "").replace(" ", "")
    if text in CUBE_AXIS_ORDERS:
        return text
    if text in CUBE_AXIS_ORDER_ALIASES:
        return CUBE_AXIS_ORDER_ALIASES[text]
    raise ValueError(
        "Unknown cube axis order. Use TYX/TXY/YTX/YXT/XTY/XYT or 123/132/213/231/312/321."
    )


def cube_axis_order_numeric_label(order: str) -> str:
    normalized = normalize_cube_axis_order(order)
    for numeric, letter_order in CUBE_AXIS_ORDER_ALIASES.items():
        if letter_order == normalized:
            return numeric
    return ""


def cube_axis_order_display_label(order: str) -> str:
    normalized = normalize_cube_axis_order(order)
    return f"{cube_axis_order_numeric_label(normalized)} | {normalized}"


def load_cube(cube_path: Path, *, axis_order: str = "TYX") -> tuple[np.ndarray, fits.Header]:
    cube_path = cube_path.expanduser().resolve()
    with fits.open(cube_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got ndim={data.ndim}")

    normalized_order = normalize_cube_axis_order(axis_order)
    data = np.transpose(data, axes=CUBE_AXIS_ORDERS[normalized_order])
    header["TDORDER"] = (normalized_order, "Input cube axis order normalized to TYX")

    return data, header


def frame_limits(frame: np.ndarray) -> tuple[float, float]:
    vmin, vmax = np.nanpercentile(frame, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(frame))
        vmax = float(np.nanmax(frame))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return 0.0, 1.0
    return float(vmin), float(vmax)


def distance(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def cut_polyline_points(cut: Cut) -> list[tuple[float, float]]:
    if str(getattr(cut, "mode", "line")) == "curve":
        raw_points = list(getattr(cut, "curve_points", []) or [])
        normalized: list[tuple[float, float]] = []
        for point in raw_points:
            try:
                normalized.append((float(point[0]), float(point[1])))
            except Exception:
                continue
        if len(normalized) >= 2:
            return normalized
    return [
        (float(cut.p0[0]), float(cut.p0[1])),
        (float(cut.p1[0]), float(cut.p1[1])),
    ]


def cut_center(cut: Cut) -> tuple[float, float]:
    points = cut_polyline_points(cut)
    if len(points) >= 2 and str(getattr(cut, "mode", "line")) == "curve":
        arc_lengths = polyline_arc_lengths(points)
        total_length = 0.0 if len(arc_lengths) == 0 else float(arc_lengths[-1])
        if total_length > 1e-9:
            return polyline_point_at_length(points, arc_lengths, 0.5 * total_length)
    return ((cut.p0[0] + cut.p1[0]) / 2.0, (cut.p0[1] + cut.p1[1]) / 2.0)


def cut_length(cut: Cut) -> float:
    if str(getattr(cut, "mode", "line")) == "curve":
        return polyline_length(cut_polyline_points(cut))
    return distance(cut.p0, cut.p1)


def cut_directed_angle_deg(cut: Cut) -> float:
    dx = cut.p1[0] - cut.p0[0]
    dy = cut.p1[1] - cut.p0[1]
    return float(math.degrees(math.atan2(dx, dy)))


def cut_display_angle_deg(cut: Cut) -> float:
    dx = cut.p1[0] - cut.p0[0]
    dy = cut.p1[1] - cut.p0[1]
    if dy < 0 or (dy == 0 and dx < 0):
        dx = -dx
        dy = -dy
    return float(math.degrees(math.atan2(dx, dy)))


def clamp_value(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def clamp_point(
    point: tuple[float, float], nx: int, ny: int
) -> tuple[float, float]:
    return (
        clamp_value(point[0], 0.0, nx - 1.0),
        clamp_value(point[1], 0.0, ny - 1.0),
    )


def rotate_point(
    point: tuple[float, float],
    center: tuple[float, float],
    angle_deg: float,
) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x = point[0] - center[0]
    y = point[1] - center[1]
    return (
        center[0] + cos_a * x - sin_a * y,
        center[1] + sin_a * x + cos_a * y,
    )


def vector_from_vertical_angle(angle_deg: float) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    return (math.sin(angle_rad), math.cos(angle_rad))


def ray_limit(
    origin: tuple[float, float],
    direction: tuple[float, float],
    nx: int,
    ny: int,
) -> float:
    ox, oy = origin
    dx, dy = direction
    limits: list[float] = []

    if abs(dx) > 1e-9:
        if dx > 0:
            limits.append((nx - 1.0 - ox) / dx)
        else:
            limits.append((0.0 - ox) / dx)

    if abs(dy) > 1e-9:
        if dy > 0:
            limits.append((ny - 1.0 - oy) / dy)
        else:
            limits.append((0.0 - oy) / dy)

    if not limits:
        return float("inf")

    return max(0.0, min(limits))


def segment_from_angle_length(
    angle_deg: float,
    length: float,
    anchor_mode: str,
    anchor_point: tuple[float, float] | None,
    center: tuple[float, float] | None,
    nx: int,
    ny: int,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    ux, uy = vector_from_vertical_angle(angle_deg)
    requested_length = max(float(length), 0.0)

    if anchor_mode == "center":
        if center is None:
            raise ValueError("center anchor requires a center point")
        max_half = min(
            ray_limit(center, (ux, uy), nx, ny),
            ray_limit(center, (-ux, -uy), nx, ny),
        )
        half = min(requested_length / 2.0, max_half)
        p0 = (center[0] - half * ux, center[1] - half * uy)
        p1 = (center[0] + half * ux, center[1] + half * uy)
    elif anchor_mode == "p0":
        if anchor_point is None:
            raise ValueError("p0 anchor requires an anchor point")
        actual_length = min(requested_length, ray_limit(anchor_point, (ux, uy), nx, ny))
        p0 = anchor_point
        p1 = (anchor_point[0] + actual_length * ux, anchor_point[1] + actual_length * uy)
        return p0, p1, actual_length
    elif anchor_mode == "p1":
        if anchor_point is None:
            raise ValueError("p1 anchor requires an anchor point")
        actual_length = min(
            requested_length, ray_limit(anchor_point, (-ux, -uy), nx, ny)
        )
        p0 = (anchor_point[0] - actual_length * ux, anchor_point[1] - actual_length * uy)
        p1 = anchor_point
        return p0, p1, actual_length
    else:
        raise ValueError(f"Unknown anchor mode: {anchor_mode}")

    return p0, p1, distance(p0, p1)


def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True

    if abs(o1) < 1e-9 and on_segment(p1, q1, p2):
        return True
    if abs(o2) < 1e-9 and on_segment(p1, q2, p2):
        return True
    if abs(o3) < 1e-9 and on_segment(q1, p1, q2):
        return True
    if abs(o4) < 1e-9 and on_segment(q1, p2, q2):
        return True

    return False


def min_segment_distance(cut_a: Cut, cut_b: Cut) -> float:
    if segments_intersect(cut_a.p0, cut_a.p1, cut_b.p0, cut_b.p1):
        return 0.0

    candidates = [
        distance_point_to_segment(cut_a.p0, cut_b.p0, cut_b.p1),
        distance_point_to_segment(cut_a.p1, cut_b.p0, cut_b.p1),
        distance_point_to_segment(cut_b.p0, cut_a.p0, cut_a.p1),
        distance_point_to_segment(cut_b.p1, cut_a.p0, cut_a.p1),
    ]
    return float(min(candidates))


def angle_difference_deg(cut_a: Cut, cut_b: Cut) -> float:
    delta = cut_display_angle_deg(cut_a) - cut_display_angle_deg(cut_b)
    return abs((delta + 90.0) % 180.0 - 90.0)


def rotate_cut(cut: Cut, angle_deg: float, nx: int, ny: int) -> None:
    center = ((cut.p0[0] + cut.p1[0]) / 2.0, (cut.p0[1] + cut.p1[1]) / 2.0)
    cut.p0 = clamp_point(rotate_point(cut.p0, center, angle_deg), nx, ny)
    cut.p1 = clamp_point(rotate_point(cut.p1, center, angle_deg), nx, ny)


def shift_cut(
    cut: Cut, dx: float, dy: float, nx: int, ny: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    min_dx = -min(cut.p0[0], cut.p1[0])
    max_dx = (nx - 1.0) - max(cut.p0[0], cut.p1[0])
    min_dy = -min(cut.p0[1], cut.p1[1])
    max_dy = (ny - 1.0) - max(cut.p0[1], cut.p1[1])

    dx = clamp_value(dx, min_dx, max_dx)
    dy = clamp_value(dy, min_dy, max_dy)

    return (
        (cut.p0[0] + dx, cut.p0[1] + dy),
        (cut.p1[0] + dx, cut.p1[1] + dy),
    )


def polyline_arc_lengths(points: list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.zeros(0, dtype=np.float64)
    lengths = np.zeros(len(points), dtype=np.float64)
    for idx in range(1, len(points)):
        lengths[idx] = lengths[idx - 1] + distance(points[idx - 1], points[idx])
    return lengths


def polyline_length(points: list[tuple[float, float]]) -> float:
    lengths = polyline_arc_lengths(points)
    return 0.0 if len(lengths) == 0 else float(lengths[-1])


def polyline_point_at_length(
    points: list[tuple[float, float]],
    arc_lengths: np.ndarray,
    s: float,
) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return (float(points[0][0]), float(points[0][1]))
    total_length = float(arc_lengths[-1]) if len(arc_lengths) else 0.0
    if total_length <= 1e-9:
        return (float(points[0][0]), float(points[0][1]))

    s = clamp_value(float(s), 0.0, total_length)
    seg_idx = int(np.searchsorted(arc_lengths, s, side="right") - 1)
    seg_idx = max(0, min(seg_idx, len(points) - 2))
    s0 = float(arc_lengths[seg_idx])
    s1 = float(arc_lengths[seg_idx + 1])
    if s1 <= s0 + 1e-9:
        return (
            float(points[seg_idx][0]),
            float(points[seg_idx][1]),
        )
    alpha = (s - s0) / (s1 - s0)
    p0 = points[seg_idx]
    p1 = points[seg_idx + 1]
    return (
        float((1.0 - alpha) * p0[0] + alpha * p1[0]),
        float((1.0 - alpha) * p0[1] + alpha * p1[1]),
    )


def polyline_tangent_at_length(
    points: list[tuple[float, float]],
    arc_lengths: np.ndarray,
    s: float,
    *,
    delta: float | None = None,
) -> tuple[float, float]:
    if len(points) < 2:
        return (0.0, 1.0)

    total_length = float(arc_lengths[-1]) if len(arc_lengths) else 0.0
    if total_length <= 1e-9:
        dx = float(points[-1][0] - points[0][0])
        dy = float(points[-1][1] - points[0][1])
        norm = math.hypot(dx, dy)
        return (0.0, 1.0) if norm <= 1e-9 else (dx / norm, dy / norm)

    step = max(
        float(delta) if delta is not None else 0.0,
        total_length / max(len(points) * 8, 16),
        1e-3,
    )
    s0 = max(0.0, float(s) - step)
    s1 = min(total_length, float(s) + step)
    if s1 <= s0 + 1e-9:
        s0 = max(0.0, float(s) - 2.0 * step)
        s1 = min(total_length, float(s) + 2.0 * step)
    p0 = polyline_point_at_length(points, arc_lengths, s0)
    p1 = polyline_point_at_length(points, arc_lengths, s1)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    norm = math.hypot(dx, dy)
    if norm > 1e-9:
        return dx / norm, dy / norm

    for idx in range(len(points) - 1):
        dx = float(points[idx + 1][0] - points[idx][0])
        dy = float(points[idx + 1][1] - points[idx][1])
        norm = math.hypot(dx, dy)
        if norm > 1e-9:
            return dx / norm, dy / norm
    return (0.0, 1.0)


def feature_axis_sample_positions(total_length: float, spacing: float) -> list[float]:
    total_length = max(float(total_length), 0.0)
    spacing = max(float(spacing), 1.0)
    if total_length <= 1e-9:
        return []
    if total_length <= spacing:
        return [0.5 * total_length]

    positions = list(np.arange(0.0, total_length + 1e-9, spacing, dtype=np.float64))
    if not positions:
        positions = [0.0]
    if total_length - float(positions[-1]) > 0.35 * spacing:
        positions.append(total_length)
    return [float(value) for value in positions]


def distance_point_to_segment(
    point: tuple[float, float],
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> float:
    px, py = point
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom == 0:
        return distance(point, p0)

    t = ((px - x0) * dx + (py - y0) * dy) / denom
    t = clamp_value(t, 0.0, 1.0)
    proj = (x0 + t * dx, y0 + t * dy)
    return distance(point, proj)


def bilinear_sample(frame: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    ny, nx = frame.shape
    valid = (xs >= 0.0) & (xs <= nx - 1.0) & (ys >= 0.0) & (ys <= ny - 1.0)
    out = np.full(xs.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return out

    xs_v = xs[valid]
    ys_v = ys[valid]

    x0 = np.floor(xs_v).astype(np.int32)
    y0 = np.floor(ys_v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)

    dx = xs_v - x0
    dy = ys_v - y0

    Ia = frame[y0, x0]
    Ib = frame[y0, x1]
    Ic = frame[y1, x0]
    Id = frame[y1, x1]

    wa = (1.0 - dx) * (1.0 - dy)
    wb = dx * (1.0 - dy)
    wc = (1.0 - dx) * dy
    wd = dx * dy

    out[valid] = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out


def line_geometry(
    cut: Cut,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float]:
    x0, y0 = cut.p0
    x1, y1 = cut.p1
    length = distance(cut.p0, cut.p1)
    if length < 1.0:
        raise ValueError("Cut is too short; use two distinct points.")

    n_samples = max(int(round(length)) + 1, 2)
    xs = np.linspace(x0, x1, n_samples, dtype=np.float32)
    ys = np.linspace(y0, y1, n_samples, dtype=np.float32)
    distances = np.linspace(0.0, length, n_samples, dtype=np.float32)
    perp = (-(y1 - y0) / length, (x1 - x0) / length)
    return xs, ys, distances, perp, length


def width_offsets_and_weights(width: int, weighting: str) -> tuple[np.ndarray, np.ndarray]:
    width = max(int(width), 1)
    offsets = np.arange(width, dtype=np.float32) - (width - 1) / 2.0

    if weighting == "gaussian" and width > 1:
        sigma = max(width / 2.355, 0.5)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2, dtype=np.float32)
    else:
        weights = np.ones(width, dtype=np.float32)

    weights /= np.sum(weights)
    return offsets, weights.astype(np.float32)


def weighted_profile(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack)
    weighted = np.where(valid, stack * weights[:, None], 0.0)
    weight_sum = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    profile = np.full(stack.shape[1], np.nan, dtype=np.float32)
    np.divide(
        weighted.sum(axis=0),
        weight_sum,
        out=profile,
        where=weight_sum > 0,
    )
    return profile


def line_geometry_from_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float]:
    length = distance(p0, p1)
    if length < 1.0:
        raise ValueError("Cut is too short; use two distinct points.")

    sample_count = (
        max(int(round(length)) + 1, 2) if n_samples is None else max(int(n_samples), 2)
    )
    xs = np.linspace(p0[0], p1[0], sample_count, dtype=np.float32)
    ys = np.linspace(p0[1], p1[1], sample_count, dtype=np.float32)
    distances = np.linspace(0.0, length, sample_count, dtype=np.float32)
    perp = (-(p1[1] - p0[1]) / length, (p1[0] - p0[0]) / length)
    return xs, ys, distances, perp, length


def sampled_cut_geometry(
    cut: Cut,
    *,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if str(getattr(cut, "mode", "line")) != "curve":
        xs, ys, distances, perp, length = line_geometry(cut)
        perp_x = np.full(xs.shape, float(perp[0]), dtype=np.float32)
        perp_y = np.full(xs.shape, float(perp[1]), dtype=np.float32)
        if n_samples is not None and int(n_samples) != len(xs):
            p0 = (float(cut.p0[0]), float(cut.p0[1]))
            p1 = (float(cut.p1[0]), float(cut.p1[1]))
            xs, ys, distances, perp, length = line_geometry_from_points(
                p0,
                p1,
                n_samples=int(n_samples),
            )
            perp_x = np.full(xs.shape, float(perp[0]), dtype=np.float32)
            perp_y = np.full(xs.shape, float(perp[1]), dtype=np.float32)
        return xs, ys, distances, perp_x, perp_y, float(length)

    points = cut_polyline_points(cut)
    if len(points) < 2:
        raise ValueError("Curved cut is too short; add more points.")
    arc_lengths = polyline_arc_lengths(points)
    total_length = 0.0 if len(arc_lengths) == 0 else float(arc_lengths[-1])
    if total_length < 1.0:
        raise ValueError("Curved cut is too short; add more separated points.")

    sample_count = (
        max(int(round(total_length)) + 1, 2)
        if n_samples is None
        else max(int(n_samples), 2)
    )
    distances = np.linspace(0.0, total_length, sample_count, dtype=np.float32)
    xs = np.empty(sample_count, dtype=np.float32)
    ys = np.empty(sample_count, dtype=np.float32)
    perp_x = np.empty(sample_count, dtype=np.float32)
    perp_y = np.empty(sample_count, dtype=np.float32)
    delta = max(total_length / max(sample_count * 4, 16), 1e-3)
    for index, distance_value in enumerate(distances):
        point = polyline_point_at_length(points, arc_lengths, float(distance_value))
        tx, ty = polyline_tangent_at_length(
            points,
            arc_lengths,
            float(distance_value),
            delta=delta,
        )
        xs[index] = float(point[0])
        ys[index] = float(point[1])
        perp_x[index] = float(-ty)
        perp_y[index] = float(tx)
    return xs, ys, distances, perp_x, perp_y, float(total_length)


def cut_point_at_distance(cut: Cut, distance_value: float) -> tuple[float, float]:
    points = cut_polyline_points(cut)
    if len(points) < 2:
        return (float(cut.p0[0]), float(cut.p0[1]))
    if str(getattr(cut, "mode", "line")) != "curve":
        length = max(distance(cut.p0, cut.p1), 1e-9)
        alpha = clamp_value(float(distance_value) / length, 0.0, 1.0)
        return (
            float((1.0 - alpha) * cut.p0[0] + alpha * cut.p1[0]),
            float((1.0 - alpha) * cut.p0[1] + alpha * cut.p1[1]),
        )
    arc_lengths = polyline_arc_lengths(points)
    total_length = 0.0 if len(arc_lengths) == 0 else float(arc_lengths[-1])
    if total_length <= 1e-9:
        return (float(points[0][0]), float(points[0][1]))
    return polyline_point_at_length(points, arc_lengths, float(distance_value))


def compute_td(
    cube: np.ndarray,
    cut: Cut,
    t_ini: int,
    t_fin: int,
    stride: int,
    width: int,
    weighting: str,
    dynamic_geometry: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    stride = max(int(stride), 1)
    t_indices = np.arange(t_ini, t_fin + 1, stride, dtype=np.int32)
    offsets, weights = width_offsets_and_weights(width, weighting)
    dynamic_geometry = dynamic_geometry or {}

    if dynamic_geometry and str(getattr(cut, "mode", "line")) == "curve":
        raise ValueError("Dynamic geometry is not available for curved cuts yet.")

    if dynamic_geometry:
        geometries = [dynamic_geometry.get(int(t), (cut.p0, cut.p1)) for t in t_indices]
        lengths = [distance(p0, p1) for p0, p1 in geometries]
        max_length = max(lengths) if lengths else cut_length(cut)
        n_samples = max(int(round(max_length)) + 1, 2)
        distances = np.linspace(0.0, max_length, n_samples, dtype=np.float32)
        td = np.empty((len(t_indices), n_samples), dtype=np.float32)

        for i, t in enumerate(t_indices):
            p0, p1 = dynamic_geometry.get(int(t), (cut.p0, cut.p1))
            frame_sample_count = max(int(round(distance(p0, p1))) + 1, 2)
            xs, ys, frame_distances, perp, _cut_length = line_geometry_from_points(
                p0, p1, n_samples=frame_sample_count
            )
            samples = []
            for offset in offsets:
                x_off = xs + offset * perp[0]
                y_off = ys + offset * perp[1]
                samples.append(bilinear_sample(cube[t], x_off, y_off))

            stack = np.stack(samples, axis=0)
            profile = weighted_profile(stack, weights)
            row = np.full(n_samples, np.nan, dtype=np.float32)
            valid = np.isfinite(profile) & np.isfinite(frame_distances)
            if np.count_nonzero(valid) == 1:
                frame_index = int(np.flatnonzero(valid)[0])
                target_index = int(
                    np.argmin(np.abs(distances - float(frame_distances[frame_index])))
                )
                row[target_index] = float(profile[frame_index])
            elif np.count_nonzero(valid) >= 2:
                src_x = np.asarray(frame_distances[valid], dtype=np.float64)
                src_y = np.asarray(profile[valid], dtype=np.float64)
                lo = float(src_x[0])
                hi = float(src_x[-1])
                inside = (distances >= lo) & (distances <= hi)
                if np.any(inside):
                    row[inside] = np.interp(
                        np.asarray(distances[inside], dtype=np.float64),
                        src_x,
                        src_y,
                    ).astype(np.float32)
            td[i] = row

        return td, {
            "t_indices": t_indices,
            "distances": distances,
            "cut_length": float(max_length),
            "dynamic": True,
            "frame_lengths": np.asarray(lengths, dtype=np.float32),
        }

    xs, ys, distances, perp_x, perp_y, cut_length_value = sampled_cut_geometry(cut)

    td = np.empty((len(t_indices), len(xs)), dtype=np.float32)

    for i, t in enumerate(t_indices):
        samples = []
        for offset in offsets:
            x_off = xs + offset * perp_x
            y_off = ys + offset * perp_y
            samples.append(bilinear_sample(cube[t], x_off, y_off))

        stack = np.stack(samples, axis=0)
        td[i] = weighted_profile(stack, weights)

    return td, {
        "t_indices": t_indices,
        "distances": distances,
        "cut_length": cut_length_value,
        "dynamic": False,
    }


def td_visual_spline(td: np.ndarray) -> np.ndarray:
    rows, cols = td.shape
    if rows < 2 or cols < 2:
        return td

    target = max(rows, cols)
    zoom_y = target / rows if rows < target else 1.0
    zoom_x = target / cols if cols < target else 1.0
    if zoom_y == 1.0 and zoom_x == 1.0:
        return td

    order = 3 if min(rows, cols) >= 4 else 1
    finite = np.isfinite(td)
    filled = np.where(finite, td, 0.0).astype(np.float32)
    weights = finite.astype(np.float32)

    filled_zoom = ndimage_zoom(
        filled, (zoom_y, zoom_x), order=order, mode="nearest", prefilter=order > 1
    )
    weights_zoom = ndimage_zoom(
        weights, (zoom_y, zoom_x), order=1, mode="nearest", prefilter=False
    )

    out = np.full(filled_zoom.shape, np.nan, dtype=np.float32)
    np.divide(filled_zoom, weights_zoom, out=out, where=weights_zoom > 1e-6)
    return out


def display_ticks(
    real_lo: float,
    real_hi: float,
    display_size: int,
    n_ticks: int = 5,
) -> tuple[np.ndarray, list[str]]:
    display_size = max(int(display_size), 2)
    tick_count = max(2, min(n_ticks, display_size))
    positions = np.linspace(0.0, display_size - 1.0, tick_count)
    labels = np.linspace(real_lo, real_hi, tick_count)
    out_labels: list[str] = []
    for value in labels:
        if abs(value - round(value)) < 1e-6:
            out_labels.append(str(int(round(value))))
        else:
            out_labels.append(f"{value:.1f}")
    return positions, out_labels


def map_value_to_display(value: float, real_lo: float, real_hi: float, display_size: int) -> float:
    display_size = max(int(display_size), 2)
    if abs(real_hi - real_lo) < 1e-9:
        return 0.5 * (display_size - 1.0)
    alpha = (value - real_lo) / (real_hi - real_lo)
    alpha = clamp_value(alpha, 0.0, 1.0)
    return alpha * (display_size - 1.0)


def cut_cache_key(cut: Cut, panel: TDPanel | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(panel, dict):
        t_ini = int(panel.get("t_ini", 0))
        t_fin = int(panel.get("t_fin", 0))
        stride = int(panel.get("stride", 1))
        width = int(panel.get("width", 1))
        weighting = str(panel.get("weighting", "uniform"))
    else:
        t_ini = int(panel.t_ini)
        t_fin = int(panel.t_fin)
        stride = int(panel.stride)
        width = int(panel.width)
        weighting = str(panel.weighting)
    return (
        cut.cut_id,
        round(cut.p0[0], 3),
        round(cut.p0[1], 3),
        round(cut.p1[0], 3),
        round(cut.p1[1], 3),
        str(getattr(cut, "mode", "line")),
        str(getattr(cut, "curve_fit", "line")),
        tuple(
            (
                round(float(point[0]), 3),
                round(float(point[1]), 3),
            )
            for point in (getattr(cut, "curve_points", []) or [])
        ),
        t_ini,
        t_fin,
        stride,
        width,
        weighting,
    )


def panel_title(panel: TDPanel, cut: Cut | None) -> str:
    if cut is None:
        return f"{panel.name} | no cut"
    return (
        f"{panel.name} | {cut.name} | "
        f"t={panel.t_ini}:{panel.t_fin}:{panel.stride} | "
        f"w={panel.width} {panel.weighting}"
    )


def make_default_panels(nt: int) -> list[TDPanel]:
    return [
        TDPanel(
            panel_id=i + 1,
            name=f"P{i + 1}",
            t_ini=0,
            t_fin=nt - 1,
            stride=1,
            width=1,
            weighting="uniform",
        )
        for i in range(MAX_PANELS)
    ]


def clamp_int(value: int, lo: int, hi: int) -> int:
    return min(max(int(value), lo), hi)
