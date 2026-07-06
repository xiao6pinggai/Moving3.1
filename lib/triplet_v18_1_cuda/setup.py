from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="triplet_topk_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="triplet_topk_cuda_ext",
            sources=[
                "triplet_topk_ext.cpp",
                "triplet_topk_ext_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
