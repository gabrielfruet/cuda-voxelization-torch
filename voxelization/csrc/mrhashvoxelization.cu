#include "ATen/core/TensorBody.h"
#include "ATen/ops/full.h"
#include "c10/cuda/CUDAGuard.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <torch/all.h>
#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL

// Simple spatial hash function for 3D coordinates
__device__ inline unsigned int hash_coords(int x, int y, int z, unsigned int size) {
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) % size;
}

__global__ void mrhash_dynamic_voxel_kernel(
    float* __restrict__ points, // [num_points, features_dim]
    uint64_t* hash_to_voxel_key_table,
    int* hash_to_voxel_index_table,
    float* __restrict__ voxel_sizes, // [vx_size, vy_size, vz_size]
    float* voxel_coords, // [num_voxels, 3] Output: voxel coordinates for voxel index (e.g voxel 3 is at voxel_coords[3*3]). 
    int* voxel_idx_per_point, // [num_points] Output: voxel index for each point
    int* voxel_count, // [1] Output: total number of unique voxels
    int max_num_voxels,  // Maximum number of voxels
    int table_size,
    int num_points,
    int features_dim
    ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    float* pt = points + tid * features_dim;
    float x = pt[0];
    float y = pt[1];
    float z = pt[2];
    // Convert point to integer grid coordinates

    float vx_size = voxel_sizes[0];
    float vy_size = voxel_sizes[1];
    float vz_size = voxel_sizes[2];

    int vx = floor(x / vx_size);
    int vy = floor(y / vy_size);
    int vz = floor(z / vz_size);

    // Encode 3D key into 64-bit for atomic comparison
    // Use unsigned casts to avoid sign extension issues with bitwise OR
    uint64_t p1 = (uint64_t)(unsigned int)vx;
    uint64_t p2 = (uint64_t)(unsigned int)vy;
    uint64_t p3 = (uint64_t)(unsigned int)vz;
    uint64_t voxel_unique_key = (p1 << 42) | (p2 << 21) | p3;
    
    unsigned int voxel_hash = hash_coords(vx, vy, vz, table_size);

    // --- STEP 1: Intra-Warp Deduplication ---
    // Find all threads in this warp that are trying to write to the SAME voxel
    unsigned int mask = __match_any_sync(0xFFFFFFFF, voxel_unique_key);
    int leader = __ffs(mask) - 1; // Elect the first thread as leader
    int lane_id = threadIdx.x % 32;

    // --- STEP 2: Linear Probing (The "Retries") ---
    int voxel_idx = -1;
    if (lane_id == leader) {
        while (true) {
            // atomicCAS: If slot is EMPTY (0), swap with our KEY.
            uint64_t old_voxel_key = atomicCAS(
                (unsigned long long*)(hash_to_voxel_key_table + voxel_hash),
                (unsigned long long)EMPTY_KEY,
                (unsigned long long)voxel_unique_key
            );

            if (old_voxel_key == EMPTY_KEY) {
                // We successfully claimed this slot.
                voxel_idx = atomicAdd(voxel_count, 1);
                if (voxel_idx < max_num_voxels) {
                    // Store the voxel index atomically to ensure visibility
                    
                    // Store coordinates
                    voxel_coords[voxel_idx * 3 + 0] = (float)vx;
                    voxel_coords[voxel_idx * 3 + 1] = (float)vy;
                    voxel_coords[voxel_idx * 3 + 2] = (float)vz;

                    __threadfence(); // Ensure voxel_coords are visible before index

                    atomicExch(hash_to_voxel_index_table + voxel_hash, voxel_idx);
                } else {
                    voxel_idx = -1; // Overflow
                }
                break;

            } else if (old_voxel_key == voxel_unique_key) {
                volatile int* idx_table = (volatile int*)hash_to_voxel_index_table;

                while ((voxel_idx = idx_table[voxel_hash]) == -1) {
                  // Avoid busy-wait heat; yield thread slightly
                  __nanosleep(10);
                }
                break;
            }

            // Collision: Move to the next slot (Linear Probing)
            voxel_hash = (voxel_hash + 1) % table_size;
        }
    }

    // --- STEP 3: Broadcast Result ---
    voxel_idx = __shfl_sync(mask, voxel_idx, leader);
    
    voxel_idx_per_point[tid] = voxel_idx;
}

namespace voxelization {
int dynamic_voxelization_cuda(const at::Tensor &points,
                                  const at::Tensor &voxel_sizes,
                                  const at::Tensor &voxel_coords,
                                  const at::Tensor &voxel_idx_per_point,
                                  int64_t max_num_voxels) {
  at::cuda::CUDAGuard guard(points.device());

  const int num_points = points.size(0);
  const int features_dim = points.size(1);

  if (num_points == 0) {
    voxel_idx_per_point.zero_();
    return 0;
  }

  if (voxel_idx_per_point.numel() != num_points) {
    throw std::runtime_error(
        "voxel_idx_per_point tensor has incorrect number of elements.");
  }

  if (voxel_sizes.numel() != 3) {
    throw std::runtime_error(
        "voxel_sizes tensor must have exactly 3 elements.");
  }

  if (voxel_coords.numel() != max_num_voxels * 3) {
    throw std::runtime_error(
        "voxel_coords tensor has incorrect number of elements.");
  }

  const int table_size = 2 * max_num_voxels; // Load factor of 0.5

  at::Tensor d_hash_to_voxel_key_table = at::full(
      {table_size}, (uint64_t)EMPTY_KEY, points.options().dtype(at::kUInt64));
  at::Tensor d_hash_to_voxel_index_table = at::full(
      {table_size}, -1, points.options().dtype(at::kInt));

  const int threads = 256;
  const int blocks = (num_points + threads - 1) / threads;

  at::Tensor d_voxel_count = at::full({1}, 0, points.options().dtype(at::kInt));

  mrhash_dynamic_voxel_kernel<<<blocks, threads>>>(
      points.data_ptr<float>(), d_hash_to_voxel_key_table.mutable_data_ptr<uint64_t>(),
      d_hash_to_voxel_index_table.mutable_data_ptr<int>(),
      voxel_sizes.mutable_data_ptr<float>(), voxel_coords.mutable_data_ptr<float>(),
      voxel_idx_per_point.mutable_data_ptr<int>(), d_voxel_count.mutable_data_ptr<int>(),
      max_num_voxels, table_size, num_points, features_dim);

  C10_CUDA_CHECK(cudaGetLastError());

  return d_voxel_count.item<int>();
}
}
