#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.widgets import Slider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a 3D FITS cube with axes interpreted as "
            "NAXIS1=x, NAXIS2=y, NAXIS3=time. In NumPy this is data[t, y, x]."
        )
    )
    parser.add_argument(
        "--cube",
        type=Path,
        default=None,
        help="Path to the FITS cube. If omitted, the inspector asks for one at startup.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Initial time index t to display.",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=None,
        help="Optional initial x pixel to mark.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=None,
        help="Optional initial y pixel to mark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG output path for the current frame.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window. Useful with --output.",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap for imshow.",
    )
    return parser.parse_args()


def _prompt_for_cube_path(initial_cube: Path | None = None) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    current_path = (
        initial_cube.expanduser()
        if initial_cube is not None
        else Path.home()
    )
    selected = filedialog.askopenfilename(
        title="Open FITS cube",
        initialdir=str(current_path.parent if current_path.suffix else current_path),
        filetypes=[("FITS", "*.fits *.fit *.fts"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    return Path(selected).expanduser().resolve() if selected else None


def load_cube(cube_path: Path) -> tuple[np.ndarray, fits.Header]:
    cube_path = cube_path.expanduser().resolve()
    with fits.open(cube_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D FITS cube, got ndim={data.ndim}")

    return data, header


def clamp_index(idx: int, size: int, name: str) -> int:
    if idx < 0 or idx >= size:
        raise ValueError(f"{name}={idx} is outside valid range [0, {size - 1}]")
    return idx


def compute_limits(frame: np.ndarray) -> tuple[float, float]:
    vmin, vmax = np.nanpercentile(frame, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(frame))
        vmax = float(np.nanmax(frame))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)


def main() -> None:
    args = parse_args()
    cube_path = args.cube.expanduser().resolve() if args.cube is not None else None
    if cube_path is None:
        cube_path = _prompt_for_cube_path()
        if cube_path is None:
            raise SystemExit("No FITS cube selected.")
    cube, header = load_cube(cube_path)
    nt, ny, nx = cube.shape

    t_index = clamp_index(args.time_index, nt, "time-index")
    x_init = None if args.x is None else clamp_index(args.x, nx, "x")
    y_init = None if args.y is None else clamp_index(args.y, ny, "y")

    print(f"Loaded cube: {cube_path}")
    print(f"Shape: nt={nt}, ny={ny}, nx={nx}")
    print("Axis convention: data[t, y, x]  <->  DS9 axes (x, y, frame/time)")
    print(
        "Header sizes: "
        f"NAXIS1={header.get('NAXIS1')}, "
        f"NAXIS2={header.get('NAXIS2')}, "
        f"NAXIS3={header.get('NAXIS3')}"
    )

    frame = cube[t_index]
    vmin, vmax = compute_limits(frame)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.16)

    im = ax.imshow(
        frame,
        origin="lower",
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Feature value")

    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    ax.set_title(
        f"Cube slice at t={t_index} | NumPy: data[t, y, x] | DS9 axes: x, y, time"
    )

    info_text = fig.text(
        0.02,
        0.02,
        "Click on the image to print x, y, t and value. "
        "Use the slider or left/right arrows to change time.",
        fontsize=10,
    )

    marker_line, = ax.plot([], [], marker="+", color="tab:red", ms=12, mew=2)

    def set_marker(x: int | None, y: int | None, t: int) -> None:
        if x is None or y is None:
            marker_line.set_data([], [])
            return

        marker_line.set_data([x], [y])
        value = float(cube[t, y, x])
        info = f"x={x}, y={y}, t={t}, value={value:.6g}"
        info_text.set_text(info)
        print(info)

    slider_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])
    t_slider = Slider(
        ax=slider_ax,
        label="time index",
        valmin=0,
        valmax=nt - 1,
        valinit=t_index,
        valstep=1,
    )

    current = {"t": t_index}
    selected = {"x": x_init, "y": y_init}

    def redraw(t: int) -> None:
        current["t"] = t
        new_frame = cube[t]
        new_vmin, new_vmax = compute_limits(new_frame)
        im.set_data(new_frame)
        im.set_clim(new_vmin, new_vmax)
        ax.set_title(
            f"Cube slice at t={t} | NumPy: data[t, y, x] | DS9 axes: x, y, time"
        )
        set_marker(selected["x"], selected["y"], t)
        fig.canvas.draw_idle()

    def on_slider_change(val: float) -> None:
        redraw(int(val))

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if 0 <= x < nx and 0 <= y < ny:
            selected["x"] = x
            selected["y"] = y
            set_marker(x, y, current["t"])
            fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key not in {"left", "right"}:
            return

        step = -1 if event.key == "left" else 1
        new_t = min(max(current["t"] + step, 0), nt - 1)
        if new_t != current["t"]:
            t_slider.set_val(new_t)

    t_slider.on_changed(on_slider_change)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    if x_init is not None and y_init is not None:
        set_marker(x_init, y_init, t_index)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved PNG: {output_path}")

    if args.no_show:
        plt.close(fig)
        return

    plt.show()


if __name__ == "__main__":
    main()
