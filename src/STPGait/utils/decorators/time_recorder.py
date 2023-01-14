from time import time
from typing import Callable

def timer(func: Callable):
    func_name = func.__name__
    
    def call(*args, **kwargs):
        start = time()
        out = func(*args, **kwargs)
        end = time()
        print(f"% TIME RECORDER: FUNC {func_name}, DURATION: {end - start}s. %", flush=True)
        return out
    
    return call