import matplotlib.pyplot as plt
import numpy as np
import torch

from voxelization.ops import _dynamic_voxelize_gpu


def visualize_voxelization():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping visualization.")
        return

    # 1. Setup parameters
    # Adjust parameters for KITTI-like point cloud
    voxel_size = [0.1, 0.1, 0.1]

    # 2. Load points from file
    try:
        # Assuming Kitti format: N x 4 float32 (x, y, z, r)
        points_np = np.fromfile("000000.bin", dtype=np.float32).reshape(-1, 4)
        points = torch.from_numpy(points_np[:, :3]).cuda().contiguous()  # Use XYZ only
        print(f"Loaded {points.shape[0]} points from 000000.bin")

        # Determine ranges from data
        min_p = points.min(dim=0)[0]
        max_p = points.max(dim=0)[0]

        print(f"Point cloud range:")
        print(f"Min: {min_p.cpu().numpy()}")
        print(f"Max: {max_p.cpu().numpy()}")

        # Add some padding to the range
        coors_range = [
            min_p[0].item(),
            min_p[1].item(),
            min_p[2].item(),
            max_p[0].item() + 0.1,
            max_p[1].item() + 0.1,
            max_p[2].item() + 0.1,
        ]

    except FileNotFoundError:
        print("000000.bin not found, running with random points.")
        # Fallback to random generation
        coors_range = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        N = 500
        points = torch.rand(N, 3, device="cuda")
        points = points * 0.8 + 0.1

    N = points.shape[0]

    # 3. Prepare output tensors
    coors = torch.full((N, 3), -1, dtype=torch.int32, device="cuda")
    point_voxel_idx = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    # 4. Run voxelization
    print("Voxelizing...")
    _dynamic_voxelize_gpu(points, coors, point_voxel_idx, voxel_size, coors_range)

    # 5. Bring data to CPU for plotting
    points_np = points.cpu().numpy()
    coors_np = coors.cpu().numpy()

    # Downsample for visualization if too many points
    max_vis_points = 11000
    if points_np.shape[0] > max_vis_points:
        print(
            f"Downsampling points from {points_np.shape[0]} to {max_vis_points} for visualization..."
        )
        choice = np.random.choice(points_np.shape[0], max_vis_points, replace=False)
        points_vis = points_np[choice]
    else:
        points_vis = points_np

    # Calculate voxel centers for visualization
    # coors is (z, y, x) indices
    # We want (x, y, z) coordinates
    # voxel_center = coors * voxel_size + min_range + half_voxel

    valid_mask = coors_np[:, 0] != -1
    valid_coors = coors_np[valid_mask]

    # Note: coors are (z, y, x), we want x, y, z for plotting
    unique_coors = np.unique(valid_coors, axis=0)

    if unique_coors.shape[0] > max_vis_points:
        print(
            f"Downsampling voxels from {unique_coors.shape[0]} to {max_vis_points} for visualization..."
        )
        choice = np.random.choice(unique_coors.shape[0], max_vis_points, replace=False)
        unique_coors = unique_coors[choice]

    voxel_centers_x = (
        unique_coors[:, 2] * voxel_size[0] + coors_range[0] + voxel_size[0] / 2
    )
    voxel_centers_y = (
        unique_coors[:, 1] * voxel_size[1] + coors_range[1] + voxel_size[1] / 2
    )
    voxel_centers_z = (
        unique_coors[:, 0] * voxel_size[2] + coors_range[2] + voxel_size[2] / 2
    )

    # 6. Plotting
    fig = plt.figure(figsize=(12, 6))

    # Plot 1: Original Points
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], s=1, c="blue", alpha=0.3
    )
    ax1.set_title(f"Original Point Cloud ({points_vis.shape[0]} pts)")
    ax1.set_xlim(coors_range[0], coors_range[3])
    ax1.set_ylim(coors_range[1], coors_range[4])
    ax1.set_zlim(coors_range[2], coors_range[5])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plot 2: Voxel Grid (Centers)
    ax2 = fig.add_subplot(122, projection="3d")
    # Using red squares to represent voxels
    ax2.scatter(
        voxel_centers_x,
        voxel_centers_y,
        voxel_centers_z,
        s=5,
        c="red",
        marker="s",
        alpha=0.5,
    )

    ax2.set_title(
        f"Voxelized (Grid Size: {voxel_size})\n({unique_coors.shape[0]} voxels)"
    )
    ax2.set_xlim(coors_range[0], coors_range[3])
    ax2.set_ylim(coors_range[1], coors_range[4])
    ax2.set_zlim(coors_range[2], coors_range[5])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig("voxelization_vis.png")
    print("Visualization saved to voxelization_vis.png")


if __name__ == "__main__":
    visualize_voxelization()
