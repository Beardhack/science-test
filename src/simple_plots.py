from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = ["#4f6f8f", "#c35d3b", "#4d8f61", "#8a5fa8", "#c08a2c", "#4c8f8f"]


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _canvas(width: int = 960, height: int = 600) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "white")
    return img, ImageDraw.Draw(img)


def _scale(value: float, vmin: float, vmax: float, start: int, end: int) -> int:
    if not np.isfinite(value):
        value = vmin
    if abs(vmax - vmin) < 1e-12:
        return int((start + end) / 2)
    return int(start + (value - vmin) / (vmax - vmin) * (end - start))


def save_line_plot(
    path: str | Path,
    series: Iterable[tuple[str, list[float], list[float]]],
    title: str,
    x_label: str,
    y_label: str,
    hline: float | None = None,
) -> None:
    series = list(series)
    xs = [x for _, sx, _ in series for x in sx]
    ys = [y for _, _, sy in series for y in sy]
    if hline is not None:
        ys.append(hline)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_y = max((ymax - ymin) * 0.08, 0.05)
    ymin -= pad_y
    ymax += pad_y
    img, draw = _canvas()
    left, right, top, bottom = 95, 900, 70, 520
    draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
    draw.text((left, 20), title, fill=(20, 20, 20), font=_font(18))
    draw.text((390, 555), x_label, fill=(20, 20, 20), font=_font(13))
    draw.text((12, 260), y_label, fill=(20, 20, 20), font=_font(13))
    for tick in np.linspace(xmin, xmax, 5):
        x = _scale(float(tick), xmin, xmax, left, right)
        draw.line([x, bottom, x, bottom + 5], fill=(40, 40, 40))
        draw.text((x - 20, bottom + 10), f"{tick:.2g}", fill=(20, 20, 20), font=_font(10))
    for tick in np.linspace(ymin, ymax, 5):
        y = _scale(float(tick), ymin, ymax, bottom, top)
        draw.line([left - 5, y, left, y], fill=(40, 40, 40))
        draw.text((28, y - 7), f"{tick:.2f}", fill=(20, 20, 20), font=_font(10))
    if hline is not None:
        y = _scale(hline, ymin, ymax, bottom, top)
        draw.line([left, y, right, y], fill=(0, 0, 0), width=1)
    for idx, (label, sx, sy) in enumerate(series):
        color = _hex(PALETTE[idx % len(PALETTE)])
        points = [(_scale(x, xmin, xmax, left, right), _scale(y, ymin, ymax, bottom, top)) for x, y in zip(sx, sy)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for point in points:
            draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=color)
        draw.rectangle([710, 80 + idx * 20, 724, 94 + idx * 20], fill=color)
        draw.text((730, 78 + idx * 20), label[:32], fill=(20, 20, 20), font=_font(10))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_histogram_overlay(
    path: str | Path,
    values_a: Iterable[float],
    values_b: Iterable[float],
    title: str,
    label_a: str,
    label_b: str,
    bins: int = 30,
) -> None:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    counts_a, edges = np.histogram(a, bins=bins, range=(0, 1))
    counts_b, _ = np.histogram(b, bins=edges)
    ymax = max(int(counts_a.max()), int(counts_b.max()), 1)
    img, draw = _canvas()
    left, right, top, bottom = 80, 900, 65, 520
    draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
    draw.text((left, 20), title, fill=(20, 20, 20), font=_font(18))
    width = (right - left) / bins
    for i in range(bins):
        x0 = int(left + i * width)
        x1 = int(left + (i + 0.44) * width)
        x2 = int(left + (i + 0.50) * width)
        x3 = int(left + (i + 0.94) * width)
        ya = _scale(float(counts_a[i]), 0, ymax, bottom, top)
        yb = _scale(float(counts_b[i]), 0, ymax, bottom, top)
        draw.rectangle([x0, ya, x1, bottom], fill=_hex(PALETTE[0]))
        draw.rectangle([x2, yb, x3, bottom], fill=_hex(PALETTE[1]))
    draw.text((395, 555), "Estimated propensity score", fill=(20, 20, 20), font=_font(13))
    draw.text((20, 265), "Patients", fill=(20, 20, 20), font=_font(13))
    draw.rectangle([710, 80, 724, 94], fill=_hex(PALETTE[0]))
    draw.text((730, 78), label_a, fill=(20, 20, 20), font=_font(11))
    draw.rectangle([710, 105, 724, 119], fill=_hex(PALETTE[1]))
    draw.text((730, 103), label_b, fill=(20, 20, 20), font=_font(11))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_love_plot(
    path: str | Path,
    labels: list[str],
    conventional: list[float],
    hdps: list[float],
    title: str,
    threshold: float = 0.10,
) -> None:
    height = max(460, 80 + 26 * len(labels))
    img, draw = _canvas(960, height)
    left, right, top, bottom = 260, 900, 60, height - 60
    xmax = max([threshold, *conventional, *[v for v in hdps if np.isfinite(v)], 0.2]) * 1.15
    draw.text((left, 20), title, fill=(20, 20, 20), font=_font(18))
    draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
    threshold_x = _scale(threshold, 0, xmax, left, right)
    draw.line([threshold_x, top, threshold_x, bottom], fill=(0, 0, 0), width=1)
    for i, label in enumerate(labels):
        y = int(top + (i + 0.5) * (bottom - top) / max(len(labels), 1))
        draw.text((20, y - 8), label[:34], fill=(20, 20, 20), font=_font(10))
        x1 = _scale(float(conventional[i]), 0, xmax, left, right)
        draw.ellipse([x1 - 4, y - 4, x1 + 4, y + 4], fill=_hex(PALETTE[0]))
        if i < len(hdps) and np.isfinite(hdps[i]):
            x2 = _scale(float(hdps[i]), 0, xmax, left, right)
            draw.rectangle([x2 - 4, y - 4, x2 + 4, y + 4], fill=_hex(PALETTE[1]))
    draw.text((430, height - 35), "Absolute standardized mean difference", fill=(20, 20, 20), font=_font(13))
    draw.rectangle([700, 80, 714, 94], fill=_hex(PALETTE[0]))
    draw.text((720, 78), "Conventional PS", fill=(20, 20, 20), font=_font(11))
    draw.rectangle([700, 105, 714, 119], fill=_hex(PALETTE[1]))
    draw.text((720, 103), "hdPS/proxy PS", fill=(20, 20, 20), font=_font(11))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_barh(path: str | Path, labels: list[str], values: list[float], title: str, x_label: str, vline: float = 1.0) -> None:
    height = max(460, 80 + 34 * len(labels))
    img, draw = _canvas(1040, height)
    left, right, top, bottom = 360, 970, 60, height - 60
    xmin = min(0.0, min(values), vline)
    xmax = max(max(values), vline) * 1.15
    draw.text((left, 20), title, fill=(20, 20, 20), font=_font(18))
    draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
    vx = _scale(vline, xmin, xmax, left, right)
    draw.line([vx, top, vx, bottom], fill=(0, 0, 0), width=1)
    bar_h = max(8, int((bottom - top) / max(len(labels), 1) * 0.55))
    for i, (label, value) in enumerate(zip(labels, values)):
        y = int(top + (i + 0.5) * (bottom - top) / max(len(labels), 1))
        x = _scale(value, xmin, xmax, left, right)
        x0 = _scale(0, xmin, xmax, left, right)
        draw.text((20, y - 8), label[:48], fill=(20, 20, 20), font=_font(10))
        draw.rectangle([min(x0, x), y - bar_h // 2, max(x0, x), y + bar_h // 2], fill=_hex(PALETTE[0]))
        draw.text((x + 5, y - 8), f"{value:.2f}", fill=(20, 20, 20), font=_font(10))
    draw.text((520, height - 35), x_label, fill=(20, 20, 20), font=_font(13))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
