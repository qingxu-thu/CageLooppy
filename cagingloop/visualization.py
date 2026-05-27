from __future__ import annotations

import numpy as np


def _axes3d(ax=None):
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib") from exc

    fig = plt.figure()
    return fig.add_subplot(111, projection="3d")


def plot_points(points, *, values=None, ax=None, title=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    ax = _axes3d(ax)
    kwargs = {"s": 12}
    if values is not None:
        kwargs["c"] = np.asarray(values, dtype=float)
        kwargs["cmap"] = "viridis"
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    if title:
        ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    return ax


def plot_caging_path(path, *, ax=None, title="Caging path"):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must be an N x 3 array")
    ax = _axes3d(ax)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="tab:red", linewidth=2)
    ax.scatter(path[:, 0], path[:, 1], path[:, 2], color="tab:red", s=10)
    if title:
        ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    return ax
