import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stderr.reconfigure(line_buffering=True)


# Default fallback architectures
fallback_archs = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
]

nvcc_args = [
    "-O3",
    "--maxrregcount=32",
    "--use_fast_math",
]


arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")


if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "bin", "nvcc")):
    if arch_list:
        # Let NVCC expand from TORCH_CUDA_ARCH_LIST at link time; or explicitly map to -gencode list.
        pass
    else:
        # Explicit multi-arch fallback (same as your current list)
        nvcc_args.extend(fallback_archs)
else:
    raise RuntimeError(
        "CUDA toolchain not found; set CUDA_HOME or use the CUDA devel image."
    )


# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {arch_list if arch_list else 'multiple architectures'}"
        print("\n" + "=" * 50)
        print(arch_info)
        print("=" * 50 + "\n")
        super().build_extensions()


setup(
    name="fused_ssim",
    packages=["fused_ssim"],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=["ssim.cu", "ext.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": CustomBuildExtension},
)

# Print again at the end of setup.py execution
final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)
