import numpy as np

from abc import ABCMeta
from abc import abstractmethod

class BaseEncoderReduction (metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, encoded_tensor: np.ndarray, weight_tensor: np.ndarray, **kwargs) -> any:
        raise NotImplementedError()

if __name__ == "__main__":
    pass
