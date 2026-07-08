from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="feature_topk_v25_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="feature_topk_v25_cuda_ext",
            sources=[
                "feature_topk_v25_ext.cpp",
                "feature_topk_v25_ext_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
