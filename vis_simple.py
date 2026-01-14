import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from vispy import app, scene
from vispy.scene import events as scene_events
from vispy.scene import visuals

from voxelization.ops import dynamic_voxelize_gpu

# NOTE: Do not name this file vispy.py, as it conflicts with the library name.


# Work around a vispy bug where SceneMouseEvent.__repr__ may call .scale on
# mouse_move events (which don't have it), leading to a TypeError that breaks
# interaction. Provide a safe default scale value instead.
def _safe_scale(self):
    return getattr(self.mouse_event, "scale", 1.0)


scene_events.SceneMouseEvent.scale = property(_safe_scale)  # type: ignore[attr-defined]


def load_kitti_points(path: str):
    """Load a KITTI binary point cloud into (N,3) points and intensities."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"File {path} does not look like KITTI format.")
    raw = raw.reshape(-1, 4)
    return raw[:, :3], raw[:, 3]


def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def jet_colors(norm: np.ndarray) -> np.ndarray:
    norm = np.clip(norm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * norm - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4 * norm - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4 * norm - 1), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def colorize(values: np.ndarray, vmin: float, vmax: float, alpha: float = 1.0):
    norm = normalize(values, vmin, vmax)
    colors = np.ones((len(values), 4), dtype=np.float32)
    colors[:, :3] = jet_colors(norm)
    colors[:, 3] = alpha
    return colors


def voxelize(points: np.ndarray, feats: np.ndarray, voxel_size=None):
    z_span = max(float(np.ptp(points[:, 2])), 1e-3)
    voxel_size = voxel_size or [0.2, 0.2, 0.2]
    # voxel_size = voxel_size or [0.2, 0.2, z_span]
    min_xyz, max_xyz = points.min(axis=0), points.max(axis=0)
    coors_range = [*min_xyz, *max_xyz]

    pts_gpu = torch.from_numpy(points).cuda().contiguous()

    voxel_coors, point_voxel_idx, _ = dynamic_voxelize_gpu(
        pts_gpu, voxel_size, coors_range
    )

    valid = point_voxel_idx >= 0
    if valid.sum() == 0:
        return None

    uniq = torch.unique(voxel_coors[valid], dim=0)
    voxel_size_t = torch.tensor(voxel_size, device=pts_gpu.device)
    offsets = torch.tensor(coors_range[:3], device=pts_gpu.device)
    centers = uniq.float() * voxel_size_t + offsets + voxel_size_t / 2.0

    ids = point_voxel_idx[valid].cpu().numpy()
    feats_valid = feats[valid.cpu().numpy()]
    points_valid = points[valid.cpu().numpy()]

    d2 = np.linalg.norm(points_valid, axis=-1)

    voxel_max = np.zeros(len(uniq), dtype=np.float32)
    np.maximum.at(voxel_max, ids, d2)

    return centers.cpu().numpy(), voxel_max, voxel_size


def add_voxels(parent, centers_np, intensities, voxel_size, vmax_for_colors=None):
    n = len(centers_np)
    if n == 0:
        print("No voxels to visualize.")
        return

    sx, sy, sz = voxel_size
    v_base = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    ) * np.array([sx, sy, sz], dtype=np.float32)

    f_base = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [4, 5, 6],
            [4, 6, 7],
        ],
        dtype=np.uint32,
    )

    offsets = (np.arange(n, dtype=np.uint32) * 8)[:, None, None]
    vertices = (centers_np[:, None, :] + v_base[None, :, :]).reshape(-1, 3)
    faces = (f_base[None, :, :] + offsets).reshape(-1, 3)

    voxel_dist = np.linalg.norm(centers_np, axis=-1) ** 0.5
    vmax = voxel_dist.max() if vmax_for_colors is None else vmax_for_colors
    v_colors = colorize(voxel_dist, vmin=0.0, vmax=vmax, alpha=0.2)
    vertex_colors = np.repeat(v_colors, 8, axis=0)

    visuals.Mesh(  # type: ignore[attr-defined]
        vertices=vertices, faces=faces, vertex_colors=vertex_colors, parent=parent
    )

    e_base = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.uint32,
    )
    edges = (e_base[None, :, :] + offsets).reshape(-1, 2)
    visuals.Line(  # type: ignore[attr-defined]
        pos=vertices,
        connect=edges,
        color=(0, 0, 0, 1),
        width=2,
        antialias=True,
        parent=parent,
    )


def clear_scene(container):
    for child in list(container.children):
        child.parent = None


def render_frame(container, bin_path: Path, canvas):
    clear_scene(container)

    points, feats = load_kitti_points(str(bin_path))
    print(f"Loaded {len(points)} points from {bin_path.name}")

    visuals.XYZAxis(parent=container)  # type: ignore[attr-defined]

    points_norm = np.linalg.norm(points, axis=-1) ** 0.5
    scatter_max = points_norm.max()
    scatter_colors = colorize(points_norm, vmin=0, vmax=scatter_max)
    visuals.Markers(parent=container, antialias=False).set_data(  # type: ignore[attr-defined]
        points, edge_color=None, face_color=scatter_colors, size=5
    )

    voxels = voxelize(points, feats)
    if voxels:
        centers_np, densities, voxel_size = voxels
        print(f"Rendering {len(centers_np)} voxels.")
        add_voxels(
            container, centers_np, densities, voxel_size, vmax_for_colors=scatter_max
        )
    else:
        print("No voxels to visualize.")

    if canvas is not None:
        canvas.update()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI .bin point clouds in a folder"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Folder containing .bin files (default: current dir)",
    )
    args = parser.parse_args()

    folder = Path(args.path)
    files = sorted(folder.glob("*.bin"))
    if not files:
        print(f"No .bin files found in {folder}")
        sys.exit(1)

    canvas = scene.SceneCanvas(
        keys="interactive", show=True, title="Point Cloud + Voxels"
    )
    view = canvas.central_widget.add_view()
    view.camera = "turntable"
    content = scene.Node(parent=view.scene)

    state = {"idx": 0}

    def show_current():
        print("\n" + "-" * 40)
        print(f"Showing {state['idx'] + 1}/{len(files)}: {files[state['idx']].name}")
        render_frame(content, files[state["idx"]], canvas)

    def on_key(event):
        key_obj = event.key
        key_name = getattr(key_obj, "name", str(key_obj)).upper()
        if key_name in ("RIGHT", "N", "SPACE", "RETURN", "ENTER"):
            state["idx"] = (state["idx"] + 1) % len(files)
            show_current()
        elif key_name in ("LEFT", "B"):
            state["idx"] = (state["idx"] - 1) % len(files)
            show_current()

    canvas.events.key_press.connect(on_key)  # type: ignore[attr-defined]

    show_current()
    print("Keys: Right/N/Space/Enter = next, Left/B = previous. Close window to quit.")
    app.run()


if __name__ == "__main__":
    main()
