"""
Build the custom PyTorch CUDA extension.

Requires: a machine with an NVIDIA GPU, the CUDA toolkit, and PyTorch
installed with matching CUDA support (check with
`python -c "import torch; print(torch.version.cuda)"` and make sure it
matches your installed CUDA toolkit's major version).

Build in-place (recommended for development):
    cd pytorch_ext
    pip install -e .

This compiles csrc/tiled_kernels.cu and csrc/binding.cpp into a Python
extension module named `vector_search_torch_cpp`, importable from
vector_search_torch.py.
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="vector_search_torch_cpp",
    ext_modules=[
        CUDAExtension(
            name="vector_search_torch_cpp",
            sources=[
                "csrc/binding.cpp",
                "csrc/tiled_kernels.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
