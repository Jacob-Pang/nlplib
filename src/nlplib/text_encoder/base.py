import numpy as np

from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Iterable

from pyutils.pickable import PickableObject
from pyutils.wrappers import FunctionWrapper

from .ragged_tensor import padded_tensor, padded_tensor_shape
from .encoder_reduction.base import BaseEncoderReduction
from .encoder_reduction.numpy import NumpyReduction

class BaseTextEncoder (PickableObject, metaclass=ABCMeta):
    """ Docstring todo
    """
    def __init__(self, encoder_reduction: BaseEncoderReduction = NumpyReduction(np.sum)):
        self.reduce_output = encoder_reduction

    def pad_inputs(self, text_inputs: Iterable, text_weights: Iterable = None) -> tuple:
        text_tensor = padded_tensor(text_inputs, pad_value='')
        
        if text_weights is None:
            weight_tensor = np.ones(padded_tensor_shape(text_inputs), dtype=float)
            weight_tensor[text_tensor == ""] = 0
        else:
            weight_tensor = padded_tensor(text_weights, pad_value=0, dtype=float)

        assert text_tensor.shape == weight_tensor.shape, f"""
            Incompatible Input Weights:
            Cannot cast <text_weights> with padded shape {weight_tensor.shape}
                    to <text_inputs> padded shape of {text_tensor.shape}.
        """

        return text_tensor, weight_tensor

    @abstractmethod
    def flat_encode(self, flat_text_tensor: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def flat_map_encode(self, text_tensor: np.ndarray, **kwargs) -> np.ndarray:
        return self.flat_encode(text_tensor.flatten()).reshape(text_tensor.shape)

    def encode(self, text_inputs: Iterable, text_weights: Iterable = None, **kwargs) -> any:
        text_tensor, weight_tensor = self.pad_inputs(text_inputs, text_weights)
        output_tensor = self.flat_map_encode(text_tensor, **kwargs)

        return self.reduce_output(output_tensor, weight_tensor, **kwargs)

if __name__ == "__main__":
    pass
