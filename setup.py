from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name="voxelization",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "voxelization._C",
            [
                "voxelization/csrc/voxelization.cpp",
                "voxelization/csrc/voxelization_cuda.cu",
            ],
            # define Py_LIMITED_API with min version 3.9 to expose only the stable
            # limited API subset from Python.h
            # extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]},
            # py_limited_api=True,
        )
    ],  # Build 1 wheel across multiple Python versions
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={
        # "bdist_wheel": {"py_limited_api": "cp39"}
    },  # 3.9 is minimum supported Python version
)
