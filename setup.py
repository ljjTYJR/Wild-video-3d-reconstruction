from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

# Set parallel jobs if not already set
if 'MAX_JOBS' not in os.environ:
    import multiprocessing
    os.environ['MAX_JOBS'] = str(min(multiprocessing.cpu_count(), 8))  # Use available cores, max 8

# Enable verbose compilation
os.environ['VERBOSE'] = '1'

# Removed parallel extension building - caused build errors

setup(
    name='droid-slam',
    packages=find_packages(),
    ext_modules=[
        # droid_backends extensions
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
        # dpvo extensions  
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            },
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')]
            ),
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp',
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={'build_ext': BuildExtension}
)