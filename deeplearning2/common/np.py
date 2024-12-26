# coding: utf-8
from common.config import GPU

if GPU:
    import cupy as cp

    # Set allocator for GPU memory
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    # Custom scatter_add implementation
    def scatter_add(array, indices, values):
        for i, idx in enumerate(indices):
            array[idx] += values[i]
        return array

    # Replace np.add.at with custom scatter_add
    cp.scatter_add = scatter_add

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
    
    np = cp  # Alias cupy as np for consistency
else:
    import numpy as np
