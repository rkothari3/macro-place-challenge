from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='density_cuda_ext',
    ext_modules=[
        CUDAExtension(
            name='density_cuda_ext',
            sources=['density_cuda_ext.cu'],
            extra_compile_args={
                'cxx':  ['-O2'],
                'nvcc': ['-O2', '--use_fast_math'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
