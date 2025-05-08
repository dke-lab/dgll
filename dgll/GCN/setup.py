#setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gcn_extension",
    ext_modules=[
        CUDAExtension(
            "gcn_extension",
            ["gcn_extension.cpp", "gcn_fused_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-arch=sm_70"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)