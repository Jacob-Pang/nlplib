import numpy as np

from .base import BaseEncoderReduction
from pyutils.wrappers import FunctionWrapper

class NumpyReduction (FunctionWrapper, BaseEncoderReduction):
    def __init__(self, numpy_function: callable, **default_kwargs):
        FunctionWrapper.__init__(self, numpy_function, **default_kwargs)

    def __call__(self, encoded_tensor: np.ndarray, weight_tensor: np.ndarray,
        **kwargs) -> np.ndarray:
        return FunctionWrapper.__call__(
            self, encoded_tensor * weight_tensor,
            axis=tuple(np.arange(1, encoded_tensor.ndim)),
            **kwargs
        )

if __name__ == "__main__":
    pass
