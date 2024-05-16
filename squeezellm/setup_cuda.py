from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name="quant_cuda",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "quant_cuda", ["quant_cuda.cpp", "quant_cuda_kernel.dp.cpp"],
            # extra_compile_args=['-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda'],
            extra_compile_args=['-fsycl'],
            extra_link_flags=['']
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
