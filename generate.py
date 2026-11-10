"""Generate skull / pumpkin toy point clouds (2D silhouettes + 8D classification set)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

DEFAULT_SEED = 42
DEFAULT_N = 1000
FEATURE_COLS = [f"F{i:02d}" for i in range(1, 9)]
CLASS_COL = "class"
CLASS_COLORS = {"skull": "#6c757d", "pumpkin": "#e67e22"}


def _ellipse_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    a: float,
    b: float,
    cy: float,
    holes: list[tuple[float, float, float]],
) -> np.ndarray:
    inside = (x / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0
    for hx, hy, hr in holes:
        inside &= (x - hx) ** 2 + (y - hy) ** 2 >= hr**2
    return inside


def _sample_ellipse_edge(
    n: int,
    rng: np.random.Generator,
    *,
    a: float,
    b: float,
    cy: float,
    holes: list[tuple[float, float, float]],
    jitter: float,
    t_lo: float = 0.0,
    t_hi: float = 2.0 * np.pi,
) -> np.ndarray:
    pts: list[list[float]] = []
    guard = 0
    while len(pts) < n and guard < 80:
        guard += 1
        need = n - len(pts)
        t = rng.uniform(t_lo, t_hi, size=need * 8)
        rad = rng.uniform(0.96, 1.015, size=need * 8)
        x = a * rad * np.cos(t)
        y = cy + b * rad * np.sin(t)
        keep = _ellipse_mask(x, y, a=a * 1.05, b=b * 1.05, cy=cy, holes=holes)
        got = np.column_stack([x[keep], y[keep]])
        if got.size:
            pts.extend(got.tolist())
    out = np.asarray(pts[:n], dtype=float) if pts else np.zeros((0, 2))
    if len(out):
        out = out + rng.normal(0.0, jitter, size=out.shape)
        out = out[_ellipse_mask(out[:, 0], out[:, 1], a=a * 1.08, b=b * 1.08, cy=cy, holes=holes)]
    return out[:n]


def _sample_circle_ring(
    n: int,
    rng: np.random.Generator,
    *,
    cx: float,
    cy: float,
    r: float,
    jitter: float,
    width: float = 0.06,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rr = r + rng.uniform(-width, width, size=n) * r
    x = cx + rr * np.cos(t)
    y = cy + rr * np.sin(t)
    out = np.column_stack([x, y])
    out += rng.normal(0.0, jitter, size=out.shape)
    return out


def _sample_ellipse_fill(
    n: int,
    rng: np.random.Generator,
    *,
    a: float,
    b: float,
    cy: float,
    holes: list[tuple[float, float, float]],
    jitter: float,
    r_max: float = 0.70,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    pts: list[list[float]] = []
    guard = 0
    while len(pts) < n and guard < 80:
        guard += 1
        need = n - len(pts)
        t = rng.uniform(0.0, 2.0 * np.pi, size=need * 8)
        rad = np.sqrt(rng.uniform(0.0, r_max**2, size=need * 8))
        x = a * rad * np.cos(t)
        y = cy + b * rad * np.sin(t)
        keep = _ellipse_mask(x, y, a=a, b=b, cy=cy, holes=holes)
        got = np.column_stack([x[keep], y[keep]])
        if got.size:
            pts.extend(got.tolist())
    out = np.asarray(pts[:n], dtype=float) if pts else np.zeros((0, 2))
    if len(out):
        out = out + rng.normal(0.0, jitter, size=out.shape)
        out = out[_ellipse_mask(out[:, 0], out[:, 1], a=a, b=b, cy=cy, holes=holes)]
    return out[:n]


def sample_skull_2d(n: int, rng: np.random.Generator, *, scale: float = 1.0) -> np.ndarray:
    """Skull silhouette"""
    cran_a, cran_b, cran_cy = 1.05 * scale, 1.15 * scale, 0.25 * scale
    jaw_a, jaw_b, jaw_cy = 0.70 * scale, 0.45 * scale, -0.85 * scale
    cran_holes = [
        (-0.38 * scale, 0.35 * scale, 0.28 * scale),
        (0.38 * scale, 0.35 * scale, 0.28 * scale),
        (0.0, -0.05 * scale, 0.12 * scale),
    ]
    jaw_holes = [(0.0, -0.75 * scale, 0.10 * scale)]

    n_outer = int(0.18 * n)
    n_sockets = int(0.12 * n)
    n_fill = n - n_outer - n_sockets

    n_cran_edge = int(0.72 * n_outer)
    n_jaw_edge = n_outer - n_cran_edge

    cran_edge = _sample_ellipse_edge(
        n_cran_edge, rng, a=cran_a, b=cran_b, cy=cran_cy, holes=cran_holes, jitter=0.012 * scale
    )
    jaw_edge = _sample_ellipse_edge(
        n_jaw_edge, rng, a=jaw_a, b=jaw_b, cy=jaw_cy, holes=jaw_holes, jitter=0.012 * scale
    )

    n_eye = n_sockets // 3
    rings = [
        _sample_circle_ring(n_eye, rng, cx=-0.38 * scale, cy=0.35 * scale, r=0.28 * scale, jitter=0.01 * scale),
        _sample_circle_ring(n_eye, rng, cx=0.38 * scale, cy=0.35 * scale, r=0.28 * scale, jitter=0.01 * scale),
        _sample_circle_ring(
            n_sockets - 2 * n_eye,
            rng,
            cx=0.0,
            cy=-0.05 * scale,
            r=0.12 * scale,
            jitter=0.01 * scale,
            width=0.10,
        ),
    ]

    n_jaw_fill = max(0, int(0.18 * n_fill))
    n_cran_fill = n_fill - n_jaw_fill
    fill = _sample_ellipse_fill(
        n_cran_fill, rng, a=cran_a, b=cran_b, cy=cran_cy, holes=cran_holes, jitter=0.014 * scale, r_max=0.92
    )
    jaw_fill = _sample_ellipse_fill(
        n_jaw_fill,
        rng,
        a=jaw_a,
        b=jaw_b,
        cy=jaw_cy,
        holes=jaw_holes,
        jitter=0.012 * scale,
        r_max=0.90,
    )

    parts = [cran_edge, jaw_edge, *rings, fill, jaw_fill]
    pts = np.vstack([p for p in parts if len(p)])
    # helpers may drop points after jitter/masking — top up until exactly n
    guard = 0
    while len(pts) < n and guard < 40:
        guard += 1
        extra = _sample_ellipse_fill(
            n - len(pts),
            rng,
            a=cran_a,
            b=cran_b,
            cy=cran_cy,
            holes=cran_holes,
            jitter=0.012 * scale,
            r_max=0.92,
        )
        if len(extra) == 0:
            break
        pts = np.vstack([pts, extra])
    if len(pts) < n:
        raise RuntimeError(f"sample_skull_2d produced {len(pts)} < {n} points")
    rng.shuffle(pts)
    return pts[:n]


def sample_pumpkin_2d(n: int, rng: np.random.Generator, *, scale: float = 1.0) -> np.ndarray:
    """Pumpkin silhouette"""
    cx = np.array([-1.05, -0.52, 0.0, 0.52, 1.05]) * scale
    cy = np.array([-0.08, 0.02, 0.08, 0.02, -0.08]) * scale
    rad = np.array([0.68, 0.80, 0.88, 0.80, 0.68]) * scale
    squash = 0.92

    def in_union(x: np.ndarray, y: np.ndarray, shrink: float = 1.0) -> np.ndarray:
        inside = np.zeros(x.shape, dtype=bool)
        for cxi, cyi, ri in zip(cx, cy, rad):
            inside |= (x - cxi) ** 2 + (y - cyi) ** 2 <= (shrink * ri) ** 2
        return inside

    n_edge = int(0.40 * n)
    n_fill = int(0.50 * n)
    n_stem = n - n_edge - n_fill

    edge: list[list[float]] = []
    while len(edge) < n_edge:
        need = n_edge - len(edge)
        i = rng.integers(0, 5, size=need * 6)
        t = rng.uniform(0.0, 2.0 * np.pi, size=need * 6)
        x = cx[i] + rad[i] * np.cos(t)
        y = (cy[i] + rad[i] * np.sin(t)) * squash
        keep = np.ones(x.shape, dtype=bool)
        for j in range(5):
            other = np.abs(cx[i] - cx[j]) > 1e-12
            d2 = (x - cx[j]) ** 2 + (y / squash - cy[j]) ** 2
            keep &= ~(other & (d2 < (0.98 * rad[j]) ** 2))
        keep &= ~((np.abs(x) < 0.14 * scale) & (y > 0.78 * scale))
        pts = np.column_stack([x[keep], y[keep]])
        if pts.size:
            edge.extend(pts.tolist())
    edge_arr = np.asarray(edge[:n_edge], dtype=float)
    edge_arr += rng.normal(0.0, 0.012 * scale, size=edge_arr.shape)

    fill: list[list[float]] = []
    while len(fill) < n_fill:
        need = n_fill - len(fill)
        x = rng.uniform((cx - rad).min(), (cx + rad).max(), size=need * 5)
        y0 = rng.uniform((cy - rad).min(), (cy + rad).max(), size=need * 5)
        keep = in_union(x, y0, shrink=0.88)
        y = y0 * squash
        keep &= ~((np.abs(x) < 0.14 * scale) & (y > 0.78 * scale))
        pts = np.column_stack([x[keep], y[keep]])
        if pts.size:
            fill.extend(pts.tolist())
    fill_arr = np.asarray(fill[:n_fill], dtype=float)
    fill_arr += rng.normal(0.0, 0.012 * scale, size=fill_arr.shape)

    s = np.linspace(0.0, 1.0, n_stem)
    stem = np.column_stack(
        [
            0.02 * scale + 0.14 * scale * (s**1.4) + rng.normal(0.0, 0.025 * scale, n_stem),
            0.72 * scale + 0.50 * scale * s + rng.normal(0.0, 0.02 * scale, n_stem),
        ]
    )
    pts = np.vstack([edge_arr, fill_arr, stem])
    rng.shuffle(pts)
    return pts


def _pair_block(sample_fn, n: int, rng: np.random.Generator, *, scale: float) -> np.ndarray:
    """Two independent 2D shapes → 4 features."""
    p1 = sample_fn(n, rng, scale=scale)
    ang = rng.uniform(-0.35, 0.35)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    p2 = sample_fn(n, rng, scale=scale * rng.uniform(0.9, 1.1)) @ R.T
    p2 += rng.normal(0.0, 0.04 * scale, size=p2.shape)
    return np.hstack([p1, p2])


def generate_df8d(
    n_per_class: int = DEFAULT_N,
    *,
    rng: np.random.Generator | None = None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    8D toy classification set.

    - skull: silhouette in F01–F04, isotropic noise in F05–F08
    - pumpkin: noise in F01–F04, silhouette in F05–F08
    """
    rng = rng or np.random.default_rng(seed)
    blocks = []

    skull_native = _pair_block(sample_skull_2d, n_per_class, rng, scale=1.0)
    skull_foreign = rng.normal(loc=0.0, scale=0.55, size=(n_per_class, 4))
    bs = pd.DataFrame(np.hstack([skull_native, skull_foreign]), columns=FEATURE_COLS)
    bs[CLASS_COL] = "skull"
    blocks.append(bs)

    pump_foreign = rng.normal(loc=0.0, scale=0.55, size=(n_per_class, 4))
    pump_native = _pair_block(sample_pumpkin_2d, n_per_class, rng, scale=1.0)
    bp = pd.DataFrame(np.hstack([pump_foreign, pump_native]), columns=FEATURE_COLS)
    bp[CLASS_COL] = "pumpkin"
    blocks.append(bp)

    df = pd.concat(blocks, ignore_index=True)
    df[CLASS_COL] = pd.Categorical(df[CLASS_COL], categories=["skull", "pumpkin"], ordered=True)
    return df


def _save_xy(path: Path, pts: np.ndarray) -> None:
    pd.DataFrame(pts, columns=["x", "y"]).to_csv(path, index=False)


def _plot_silhouette(path: Path, pts: np.ndarray, *, color: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=14,
        c=color,
        edgecolors="white",
        linewidths=0.2,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_box_aspect(1)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="points per class / silhouette")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Independent 2D showcases (separate streams so regenerating 8D does not shift them)
    skull = sample_skull_2d(args.n, np.random.default_rng(args.seed + 1))
    pumpkin = sample_pumpkin_2d(args.n, np.random.default_rng(args.seed + 2))
    _save_xy(DATA_DIR / "skull_2d.csv", skull)
    _save_xy(DATA_DIR / "pumpkin_2d.csv", pumpkin)

    df8 = generate_df8d(args.n, seed=args.seed)
    df8.to_csv(DATA_DIR / "df8d.csv", index=False)

    print(f"wrote {DATA_DIR / 'skull_2d.csv'}  shape={skull.shape}")
    print(f"wrote {DATA_DIR / 'pumpkin_2d.csv'}  shape={pumpkin.shape}")
    print(f"wrote {DATA_DIR / 'df8d.csv'}  shape={df8.shape}")
    print(df8[CLASS_COL].value_counts().to_string())

    if not args.no_figures:
        _plot_silhouette(FIG_DIR / "skull.png", skull, color=CLASS_COLORS["skull"], title="skull (2D)")
        _plot_silhouette(FIG_DIR / "pumpkin.png", pumpkin, color=CLASS_COLORS["pumpkin"], title="pumpkin (2D)")
        print(f"wrote {FIG_DIR / 'skull.png'}")
        print(f"wrote {FIG_DIR / 'pumpkin.png'}")


if __name__ == "__main__":
    main()
