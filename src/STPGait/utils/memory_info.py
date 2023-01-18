import psutil
from typing import Callable

import numpy as np

def get_chunk_size(data: np.ndarray, op: Callable[[int], int] = lambda x: x, take_integer: bool = True) -> int:
    """ It returns chunk size suitable to prevent memory leakage. This is useful when data is huge
    and an operation should be performed into several sub data parts.

    Args:
        data (np.ndarray): _description_
        op (Callable[[int], int], optional): operation to apply on data size. Defaults to identity operation.
        take_integer (bool, optional): If True, then return chunk size as integer, O.W. as float.
    Returns:
        int: Chunk size
    """
    mem_bytes = psutil.virtual_memory().total
    if data.dtype in [np.uint8, np.int8]:
        num_bytes = 1
    elif data.dtype in [np.float16, np.uint16, np.int16]:
        num_bytes = 2
    elif data.dtype in [np.float32, np.int32, np.uint32]:
        num_bytes = 4
    elif data.dtype in [np.float64, np.int64, np.uint64]:
        num_bytes = 8
    
    data_bytes = op(data.size) * num_bytes
    chunks = data_bytes / mem_bytes
    if take_integer:
        chunks = max(1, chunks)
    print(f"@ Data Chunking: Available Memory {mem_bytes}B, Data usage {data_bytes}B, Chunk size {chunks} @", flush=True)

    return chunks    