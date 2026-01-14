#include "ATen/core/TensorBody.h"
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

namespace voxelization {

void dynamic_voxelize_gpu(const at::Tensor &points, at::Tensor &coors,
                          const at::Tensor &point_voxel_idx,
                          const std::vector<double> voxel_size,
                          const std::vector<double> coors_range,
                          const int64_t NDim = 3);

  // Defines the operators
TORCH_LIBRARY(voxelization, m) {
  m.def("dynamic_voxelize_gpu(Tensor points, Tensor coors, Tensor point_voxel_idx, "
        "float[] voxel_size, float[] coors_range, int NDim) -> ()");
}

TORCH_LIBRARY_IMPL(voxelization, CUDA, m) {
  m.impl("dynamic_voxelize_gpu", dynamic_voxelize_gpu);
}

} // namespace voxelization

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}