#include "c10/core/ScalarType.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <torch/types.h>
#include <thrust/sort.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

__global__ void identify_starts_kernel(int *sorted_voxel_indices,
                                       int *head_mask, int num_points) {
  CUDA_1D_KERNEL_LOOP(idx, num_points) {
    if (idx == 0) {
      head_mask[0] = sorted_voxel_indices[0] == -1 ? -1: 0;
      continue;
    }

    head_mask[idx] = (sorted_voxel_indices[idx] != sorted_voxel_indices[idx - 1]);
  }
}

__global__ void flatten_index_kernel(const int *coors, int *flat_indices,
                                     int grid_x, int grid_y, int num_points) {
  CUDA_1D_KERNEL_LOOP(idx, num_points) {
    int x = coors[idx * 3 + 0];

    if (x == -1) {
      flat_indices[idx] = -1;
      continue;
    }

    int y = coors[idx * 3 + 1];
    int z = coors[idx * 3 + 2];
    flat_indices[idx] = z + y * grid_x + x * (grid_x * grid_y);
  }
}

  namespace voxelization {

  void dynamic_voxelize_gpu(const at::Tensor &points, at::Tensor &coors,
                            const at::Tensor &point_voxel_idx,
                            const std::vector<double> voxel_size,
                            const std::vector<double> coors_range,
                            const int64_t NDim = 3) {
    // current version tooks about 0.04s for one frame on cpu
    // check device
    CHECK_INPUT(points);

    at::cuda::CUDAGuard device_guard(points.device());

    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const double voxel_x = voxel_size[0];
    const double voxel_y = voxel_size[1];
    const double voxel_z = voxel_size[2];
    const double coors_x_min = coors_range[0];
    const double coors_y_min = coors_range[1];
    const double coors_z_min = coors_range[2];
    const double coors_x_max = coors_range[3];
    const double coors_y_max = coors_range[4];
    const double coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    const int col_blocks = at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
    dim3 blocks(col_blocks);
    dim3 threads(threadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", [&] {
      dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
          points.contiguous().data_ptr<scalar_t>(),
          coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
          coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
          coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
    });
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    // Compute flat index: z + y*grid_x + x*grid_x*grid_y
    // Avoiding mm/mv on Int tensors as it may not be supported on CUDA
    auto flat_unique_voxel_index = at::empty_like(point_voxel_idx);
    flatten_index_kernel<<<blocks, threads, 0, stream>>>(
        coors.data_ptr<int>(), flat_unique_voxel_index.data_ptr<int>(),
        grid_x, grid_y, num_points);

    auto point_index_keys =
        at::arange(0, num_points, points.options().dtype(at::kInt));

    thrust::sort_by_key(thrust::device, flat_unique_voxel_index.data_ptr<int>(),
                        flat_unique_voxel_index.data_ptr<int>() +
                            (int)num_points,
                        point_index_keys.data_ptr<int>());
    AT_CUDA_CHECK(cudaGetLastError());


    auto head_mask = at::zeros({num_points}, points.options().dtype(at::kInt));

    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "identify_starts_kernel", [&] {
      identify_starts_kernel<<<blocks, threads, 0, stream>>>(
          flat_unique_voxel_index.data_ptr<int>(),
          head_mask.data_ptr<int>(), num_points);
    });

    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    auto sorted_point_to_voxel_idx = point_voxel_idx.clone();

    thrust::inclusive_scan(thrust::device, head_mask.data_ptr<int>(),
                           head_mask.data_ptr<int>() + num_points,
                           sorted_point_to_voxel_idx.data_ptr<int>());

    thrust::scatter(thrust::device,
                    sorted_point_to_voxel_idx.data_ptr<int>(),
                    sorted_point_to_voxel_idx.data_ptr<int>() + num_points,
                    point_index_keys.data_ptr<int>(),
                    point_voxel_idx.data_ptr<int>());


    AT_CUDA_CHECK(cudaGetLastError());


    // identify the starts

    return;
  }

  } // namespace voxelization