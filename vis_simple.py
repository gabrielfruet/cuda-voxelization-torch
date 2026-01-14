import sys

import numpy as np
import torch
from vispy import app, scene
from vispy.visuals.transforms import STTransform

from voxelization.ops import dynamic_voxelize_gpu

# NOTE: Do not name this file vispy.py, as it conflicts with the library name.


def main():
    # 1. Load Point Cloud
    file_path = "000000.bin"
    try:
        # KITTI format: x, y, z, intensity (float32)
        print(f"Loading {file_path}...")
        raw_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        points = raw_data[:, :3]  # xyz
        intensities = raw_data[:, 3]  # Intensity
        print(f"Loaded {len(points)} points.")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    # 2. Setup Vispy Canvas
    canvas = scene.SceneCanvas(
        keys="interactive", show=True, title="Simple Point Cloud Visualization"
    )
    view = canvas.central_widget.add_view()
    view.camera = "turntable"  # Options: 'turntable', 'fly', 'arcball'

    # 3. Create Scatter Plot Visual
    # Color points based on Z-height (simple coloring)
    z = points[:, 2]
    # Simple normalization for color: Map [-3, 3] meters to [0, 1]
    z_min, z_max = -5.0, 3.0
    z_norm = np.clip((z - z_min) / (z_max - z_min), 0, 1)

    # White-ish color varying with height
    colors = np.ones((len(points), 4), dtype=np.float32)
    colors[:, 0] = z_norm  # Red channel
    colors[:, 1] = z_norm * 0.5  # Green channel
    colors[:, 2] = 1.0 - z_norm  # Blue channel

    scatter = scene.visuals.Markers(parent=view.scene, antialias=False)
    scatter.set_data(points, edge_color=None, face_color=colors, size=5)

    # 4. Add Axes for reference
    scene.visuals.XYZAxis(parent=view.scene)

    tensor_pts = torch.from_numpy(points).cuda().contiguous()

    # Auto-compute coors_range from data to ensure we see something
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    # Add a bit of padding
    coors_range = [
        min_xyz[0],
        min_xyz[1],
        min_xyz[2],
        max_xyz[0],
        max_xyz[1],
        max_xyz[2],
    ]

    voxel_size = [0.2, 0.2, max_xyz[2] - min_xyz[2]]
    # voxel_size = [0.2, 0.2, 0.2]

    print(f"Using computed coors_range: {coors_range}")

    # Set max_voxels to number of points to avoid buffer overflow
    max_voxels = tensor_pts.shape[0]

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    voxel_coors, point_voxel_idx = dynamic_voxelize_gpu(
        tensor_pts, voxel_size, coors_range
    )

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Voxelization completed in {elapsed_time_ms:.3f} ms.")

    # Filter invalid voxels and get unique ones
    # valid_mask is True where NO coordinate is -1
    valid_mask = ~(voxel_coors == -1).any(dim=1)

    print(f"Valid voxels: {valid_mask.sum().item()} / {len(voxel_coors)}")
    if valid_mask.sum() == 0:
        print("Warning: No valid voxels found!")

    unique_voxel_coors = torch.unique(voxel_coors[valid_mask], dim=0)

    num_voxels = unique_voxel_coors.shape[0]

    # show voxel_coors voxels
    print(f"Number of unique voxels: {num_voxels}")

    # --- Compute Max Intensity per Voxel ---
    # 1. Get valid CPU data
    valid_mask_cpu = valid_mask.cpu().numpy()
    # point_voxel_idx contains indices into the unique set of voxels [0, num_voxels-1]
    point_voxel_ids_cpu = point_voxel_idx[valid_mask].cpu().numpy()
    valid_intensities = intensities[valid_mask_cpu]

    voxel_max_intensities = np.zeros(num_voxels, dtype=np.float32)
    voxel_densities = np.zeros(num_voxels, dtype=np.int32)
    # Aggregate max intensity
    np.maximum.at(voxel_max_intensities, point_voxel_ids_cpu, valid_intensities)
    np.add.at(voxel_densities, point_voxel_ids_cpu, 1)
    print(f"Computed max intensities for {num_voxels} voxels.")

    # Show voxels as markers
    # Markers are much faster than creating individual Box objects

    # No need to downsample for Markers (can handle millions of points)

    # Calculate voxel centers for all unique voxels
    voxel_size_t = torch.tensor(voxel_size, device=tensor_pts.device)
    coors_offset = torch.tensor(coors_range[0:3], device=tensor_pts.device)

    # Convert integer coords to world centers
    voxel_centers = (
        unique_voxel_coors.float() * voxel_size_t + coors_offset + voxel_size_t / 2.0
    )

    # Move to CPU for Vispy
    voxel_centers_np = voxel_centers.cpu().numpy()

    n_voxels = len(voxel_centers_np)
    print(f"Visualizing {n_voxels} voxels as mesh cubes.")

    if n_voxels > 0:
        # Generate Cube Mesh Data
        sx, sy, sz = voxel_size[0], voxel_size[1], voxel_size[2]

        # Base cube vertices (centered at 0)
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

        # Base cube faces (indices)
        f_base = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom
                [0, 1, 5],
                [0, 5, 4],  # Front
                [1, 2, 6],
                [1, 6, 5],  # Right
                [2, 3, 7],
                [2, 7, 6],  # Back
                [3, 0, 4],
                [3, 4, 7],  # Left
                [4, 5, 6],
                [4, 6, 7],  # Top
            ],
            dtype=np.uint32,
        )

        # Expand to all voxels (Broadcasting)
        # Vertices: (N, 8, 3) -> Reshape (N*8, 3)
        vertices = (
            voxel_centers_np[:, np.newaxis, :] + v_base[np.newaxis, :, :]
        ).reshape(-1, 3)

        # Faces: (N, 12, 3) -> Reshape (N*12, 3)
        # Add offset to indices: 0 for first voxel, 8 for second, etc.
        offsets = np.arange(n_voxels, dtype=np.uint32) * 8
        faces = (f_base[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]).reshape(
            -1, 3
        )

        # --- Color voxels by Max Intensity (Same color space as points) ---
        # Normalize Intensity [0, 1] (Assuming KITTI intensity is 0-1)
        # If intensity > 1, clip.
        # v_int = np.clip(voxel_max_intensities, 0, 1)  # Norm factor
        v_int = (voxel_densities / voxel_densities.max()) ** (0.3)  # Density normalized

        # Use same colormap logic as points (Red->Blueish)
        # Point logic for Z was: [z_norm, z_norm*0.5, 1-z_norm]
        # We reuse this logic but with intensity

        v_colors = np.zeros((n_voxels, 4), dtype=np.float32)
        v_colors[:, 0] = v_int  # Red
        v_colors[:, 1] = v_int * 0.5  # Green
        v_colors[:, 2] = 1.0 - v_int  # Blue
        v_colors[:, 3] = 0.4  # Alpha

        # Expand to vertices (8 per voxel)
        vertex_colors = np.repeat(v_colors, 8, axis=0)

        # Create Mesh
        # Using flat shading or just solid color. edge_color makes individual voxels distinct.
        voxel_mesh = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            parent=view.scene,
        )

        # Create Wireframe (Edges) for the cubes
        # Define the 12 edges of a cube (indices into 0-7 vertices)
        e_base = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # Bottom
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # Top
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # Pillars
            ],
            dtype=np.uint32,
        )  # Shape (12, 2)

        # Expand edges indices for all voxels
        # Shape (N, 12, 2) + offsets -> (N*12, 2)
        edges_connect = (
            e_base[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
        ).reshape(-1, 2)

        # Create Line Visual
        # We reuse the same 'vertices' array as the positions.
        voxel_wireframe = scene.visuals.Line(
            pos=vertices,
            connect=edges_connect,
            color=(0.0, 0, 0, 1),  # Dark red/blackish
            width=3,
            antialias=True,
            parent=view.scene,
        )
    else:
        print("No voxels to visualize.")

    # 5. Run Application
    print("Running visualization. Press 'Interaction' keys to move camera.")
    app.run()


if __name__ == "__main__":
    main()
