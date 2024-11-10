from typing import List

def validate_params(sizes_of_conv_kernels: List[int]):
    for kernel_size in sizes_of_conv_kernels:
        if kernel_size % 2 != 0:
            raise ValueError("all kernel sizes must be even")