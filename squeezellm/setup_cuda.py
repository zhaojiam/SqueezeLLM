from setuptools import Extension, setup
from torch.utils import cpp_extension
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name="quant_cuda",
    ext_modules=[
        DPCPPExtension(
            "quant_cuda", ["quant_cuda.cpp", "quant_cuda_kernel.dp.cpp"],
            include_dirs=ipex.xpu.cpp_extension.include_paths(),
        )
    ],
    cmdclass={"build_ext": DpcppBuildExtension},
)
