from typing import List

import torch
from torch import Tensor


def _dynamic_voxelize_gpu(
    points: Tensor,
    voxel_coors: Tensor,
    point_voxel_idx: Tensor,
    voxel_size: List[float],
    coors_range: List[float],
    NDim: int = 3,
) -> None:
    """
    Args:
        points: (N, 3) float tensor of points
        coors: (N, 3) int32 tensor to be filled with voxel coordinates (z, y, x) for each point
        point_voxel_idx: (N,) int32 tensor to be filled with unique voxel index for each point
        voxel_size: [v_x, v_y, v_z]
        coors_range: [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    torch.ops.voxelization.dynamic_voxelize_gpu(
        points, voxel_coors, point_voxel_idx, voxel_size, coors_range, NDim
    )


def dynamic_voxelize_gpu(
    points: Tensor,
    voxel_size: List[float],
    coors_range: List[float],
) -> tuple[Tensor, Tensor, int]:
    """
    Args:
        points: (N, 3) float tensor of points
        voxel_size: [v_x, v_y, v_z]
        coors_range: [min_x, min_y, min_z, max_x, max_y, max_z]
        max_voxels: maximum number of unique voxels to consider

    Returns:
        voxel_coors: (N, 3) int32 tensor of voxel coordinates (z, y, x) for each point
        point_voxel_idx: (N,) int32 tensor of zero-based voxel ids per point.
            Invalid/out-of-range points are set to -1. Valid voxel ids are contiguous
            (0..num_unique_voxels-1) thanks to the exclusive scan in the CUDA path.
        num_unique_voxels: integer count of unique voxels containing at least one point
    """
    N = points.shape[0]
    # voxel_coors must be size (N, 3) to hold coordinates for each point
    voxel_coors = torch.full((N, 3), -1, dtype=torch.int32, device=points.device)
    point_voxel_idx = torch.full((N,), -1, dtype=torch.int32, device=points.device)

    torch.ops.voxelization.dynamic_voxelize_gpu(
        points, voxel_coors, point_voxel_idx, voxel_size, coors_range, 3
    )

    num_unique_voxels = int(point_voxel_idx.max().item() + 1)

    # inverse coors since its zyx
    voxel_coors = voxel_coors[:, [2, 1, 0]]

    return voxel_coors, point_voxel_idx, num_unique_voxels


if __name__ == "__main__":
    # Test Case 1: Simple correctness check
    print("Running correctness check...")
    voxel_size = [0.1, 0.1, 0.1]
    coors_range = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]

    # Define points:
    # p0 and p1 are in the same voxel (0,0,0) -> c_x=0, c_y=0, c_z=0
    # p2 is in voxel (1,0,0) -> c_x=1, c_y=0, c_z=0
    points = torch.tensor(
        [
            [0.05, 0.05, 0.05],  # (0,0,0)
            [0.06, 0.06, 0.06],  # (0,0,0)
            [0.15, 0.05, 0.05],  # (1,0,0)
        ],
        device="cuda",
        dtype=torch.float32,
    )

    # Output tensors
    # coors must be (N, 3) int32
    coors = torch.full((points.shape[0], 3), -1, dtype=torch.int32, device="cuda")
    # point_voxel_idx must be (N) int32
    point_voxel_idx = torch.full(
        (points.shape[0],), -1, dtype=torch.int32, device="cuda"
    )

    _dynamic_voxelize_gpu(points, coors, point_voxel_idx, voxel_size, coors_range)

    print(f"Points:\n{points}")
    print(f"Coors (z, y, x):\n{coors}")
    print(f"Voxel Indices per point: {point_voxel_idx}")

    # Verification
    # Check coords
    # p0: 0,0,0
    expected_0 = torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
    if not torch.all(torch.eq(coors[0], expected_0)):
        print(f"FAIL: p0 coords mismatch. Got {coors[0]}, expected {expected_0}")
    else:
        print("PASS: p0 coords match")

    # p2: (0, 0, 1) corresponding to (z=0, y=0, x=1)
    expected_2 = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    if not torch.all(torch.eq(coors[2], expected_2)):
        print(f"FAIL: p2 coords mismatch. Got {coors[2]}, expected {expected_2}")
    else:
        print("PASS: p2 coords match")

    # Check Indices
    # p0 and p1 should have SAME index
    if point_voxel_idx[0] != point_voxel_idx[1]:
        print(
            f"FAIL: p0 and p1 should be in same voxel. Indices: {point_voxel_idx[0]}, {point_voxel_idx[1]}"
        )
    else:
        print("PASS: p0 and p1 have same voxel index")

    # p2 should have DIFFERENT index
    if point_voxel_idx[0] == point_voxel_idx[2]:
        print(
            f"FAIL: p0 and p2 should be in different voxels. Indices: {point_voxel_idx[0]}, {point_voxel_idx[2]}"
        )
    else:
        print("PASS: p0 and p2 have different voxel indices")

    print("------------------------------------------------")
    print("Test Case 2: Using High-Level Function & Larger Data")

    # 2. Random data test
    N = 1000
    points_rand = torch.rand((N, 3), device="cuda", dtype=torch.float32) * 10.0

    print(f"Testing with {N} random points...")

    v_coors, v_inds, _ = dynamic_voxelize_gpu(points_rand, voxel_size, coors_range)

    valid_mask = ~(v_coors == -1).any(dim=1)
    unique_coors = torch.unique(v_coors[valid_mask], dim=0)

    print(f"Total points: {N}")
    print(f"Valid mapped points: {valid_mask.sum().item()}")
    print(f"Unique voxels found: {unique_coors.shape[0]}")

    if valid_mask.sum() == 0:
        print("FAIL: No valid points found in random test!")
    else:
        print("PASS: Random test produced valid voxels.")
    if not torch.all(torch.eq(coors[2], expected_2)):
        print(f"FAIL: p2 coords mismatch. Got {coors[2]}, expected {expected_2}")

    # Check indices logic
    idx0 = point_voxel_idx[0].item()
    idx1 = point_voxel_idx[1].item()
    idx2 = point_voxel_idx[2].item()

    if idx0 == idx1 and idx0 != idx2:
        print("SUCCESS: Voxel indices logic seems correct (p0==p1 != p2)")
    else:
        print(f"FAIL: Indices logic error. {idx0}, {idx1}, {idx2}")

    # Test Case 2: Random points
    print("\nRunning random points check...")
    N = 100000
    points = torch.rand(N, 3, device="cuda") * 10.0  # scale to fit range
    coors = torch.zeros((N, 3), dtype=torch.int32, device="cuda")
    point_voxel_idx = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    start_ns = torch.cuda.Event(enable_timing=True)
    start_ns.record()
    _dynamic_voxelize_gpu(points, coors, point_voxel_idx, voxel_size, coors_range)
    end_ns = torch.cuda.Event(enable_timing=True)
    end_ns.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_ns.elapsed_time(end_ns)
    print(f"Voxelization of {N} random points took {elapsed_time_ms:.3f} ms")

    print(f"First 10 voxel indices: {point_voxel_idx[:10]}")
    if (point_voxel_idx == -1).any():
        print(
            "WARNING: Some points have index -1 (might be out of range if not covered by coors_range)"
        )
    else:
        print("SUCCESS: All random points processed validly.")

    # Test Case 3: Alignment Check
    print("------------------------------------------------")
    print("Test Case 3: Alignment Check (unsorted inputs)")

    # p_a: (1,1,1) -> (10,10,10)
    # p_b: (0,0,0) -> (0,0,0)
    points_mix = torch.tensor(
        [
            [1.05, 1.05, 1.05],  # p_a -> (10,10,10) (assuming 0.1 voxel size)
            [0.05, 0.05, 0.05],  # p_b -> (0,0,0)
        ],
        device="cuda",
        dtype=torch.float32,
    )

    v_coors_mix, v_inds_mix, _ = dynamic_voxelize_gpu(
        points_mix, voxel_size, coors_range
    )

    print(f"Points Mix:\n{points_mix}")
    print(f"Coors Mix:\n{v_coors_mix}")

    # Expected v_coors[0] is (10,10,10)
    expected_a = torch.tensor(
        [1, 1, 1], dtype=torch.int32, device="cuda"
    )  # Wait, 1.05/0.1 = 10.
    # Actually, 1.05 is index 10 if range starts at 0.
    # index = floor((1.05 - 0)/0.1) = 10.
    # Wait, my manual calc: 1.05 is 10.5 voxel units. floor is 10.

    # Let's check what we get.
    # If sorted: (0,0,0) then (10,10,10).
    # v_coors[0] would be (0,0,0) -> NOT ALIGNED with p_a.

    if v_coors_mix[0, 0] == 0:
        print(
            "FAIL: v_coors[0] is (0,0,0) but points[0] is at (1,1,1). ALIGNMENT BROKEN."
        )
        print("This confirms inplace sorting scrambles point-to-coord mapping.")
    else:
        print("PASS: v_coors[0] seems aligned.")
