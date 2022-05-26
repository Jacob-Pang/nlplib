import tensorflow as tf

from .base import BaseEncoderReduction
from pyutils.wrappers import FunctionWrapper

class TensorflowReduction (FunctionWrapper, BaseEncoderReduction):
    def __init__(self, tensorflow_function: callable, **default_kwargs):
        FunctionWrapper.__init__(self, tensorflow_function, **default_kwargs)

    def __call__(self, encoded_tensor: tf.Tensor, weight_tensor: tf.Tensor,
        **kwargs) -> tf.Tensor:
        return FunctionWrapper.__call__(
            self, encoded_tensor * weight_tensor,
            axis=tf.range(1, encoded_tensor.shape.rank),
            **kwargs
        )

# Functions
def reduce_self_weighted_average(input_tensor: tf.Tensor, axis: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(
        tf.sign(input_tensor) * tf.pow(
            input_tensor, tf.constant(2, dtype=input_tensor.dtype)
        ), axis=axis
    ) / ( # perturbation to prevent divide by zero-errors
        tf.reduce_sum(tf.abs(input_tensor), axis=axis) +
        tf.constant(0.00001, dtype=input_tensor.dtype)
    )

if __name__ == "__main__":
    pass
