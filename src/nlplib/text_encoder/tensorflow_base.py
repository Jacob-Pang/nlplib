import pandas as pd
import numpy as np
import tensorflow as tf

from abc import abstractmethod
from tqdm import tqdm
from collections.abc import Iterable
from official.nlp import optimization
from tensorflow.keras.losses import MeanSquaredError

from .base import BaseTextEncoder
from .encoder_reduction.tensorflow import TensorflowReduction

def adamw_optimizer(num_train_steps: int, warmup_ratio: float = .1,
    init_learning_rate: float = 3e-5):
    """
    Notes
        - Compatible only with tensorflow.float32 dtype
    """
    return optimization.create_optimizer(
        init_lr=init_learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=int(warmup_ratio * num_train_steps),
        optimizer_type="adamw"
    )

class BaseTensorflowTextEncoder (BaseTextEncoder): 
    def __init__(self, encoder_reduction: TensorflowReduction = TensorflowReduction(tf.reduce_sum),
        dtype: type = tf.float32) -> None:

        super().__init__(encoder_reduction)
        self.dtype = dtype

    @abstractmethod
    def trainable_variables(self):
        raise NotImplementedError()

    def pad_inputs(self, text_inputs: Iterable, text_weights: Iterable = None) -> tuple:
        text_tensor, weight_tensor = super().pad_inputs(text_inputs, text_weights)

        return tf.convert_to_tensor(text_tensor, dtype=tf.dtypes.string), \
                tf.convert_to_tensor(weight_tensor, dtypes=self.dtype)

    @abstractmethod
    def flat_encode(self, flat_text_tensor: tf.Tensor, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    def flat_map_encode(self, text_tensor: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.reshape(
            self.flat_encode(tf.reshape(text_tensor, [-1]), **kwargs),
            text_tensor.shape
        )

    def encode(self, text_inputs: Iterable, text_weights: Iterable = None, **kwargs) -> any:
        return super().encode(text_inputs, text_weights, **kwargs).numpy()

    @tf.function
    def train_on_batch(self, output_tensor: tf.Tensor, text_tensor: tf.Tensor, weight_tensor: tf.Tensor,
        optimizer, loss_fn: callable, **kwargs) -> tf.Tensor:

        with tf.GradientTape() as tape:
            current_batch_loss = loss_fn(output_tensor, self.reduce_output(self.flat_map_encode(
                        text_tensor, **kwargs), weight_tensor))
            
            gradients = tape.gradient(current_batch_loss, self.trainable_variables())
            optimizer.apply_gradients(zip(gradients, self.trainable_variables()))

        return current_batch_loss

    def train(self, encoded_outputs: np.ndarray, text_inputs: Iterable,  text_weights: Iterable,
        batch_size: int = 4, epochs: int = 100, optimizer = None, evaluation_split: float = 0.,
        loss_fn: callable = MeanSquaredError(), **kwargs) -> pd.DataFrame:

        if optimizer is None: # Use default AdamW optimizer
            num_train_steps = epochs * encoded_outputs.shape[0]
            optimizer=adamw_optimizer(num_train_steps=num_train_steps)

        # Converting inputs and outputs to tensors
        text_tensor, weight_tensor = self.pad_inputs(text_inputs, text_weights)
        output_tensor = tf.convert_to_tensor(encoded_outputs, dtype=self.dtype)

        dataset_size = encoded_outputs.shape[0]
        dataset_idx = tf.range(0, dataset_size)
        training_loss = []

        for epoch in range(epochs):
            shuffled_idx = tf.random.shuffle(dataset_idx)
            training_batch_loss = []

            for frm_idx in tqdm(range(0, dataset_size, batch_size), desc=f"Epoch{epoch + 1:>5}/{epochs:>5}"):
                to_idx = min(frm_idx + batch_size, dataset_size)

                training_batch_loss.append(
                    self.train_on_batch(
                        tf.gather(output_tensor, shuffled_idx[frm_idx:to_idx], axis=0),
                        tf.gather(text_tensor,   shuffled_idx[frm_idx:to_idx], axis=0),
                        tf.gather(weight_tensor, shuffled_idx[frm_idx:to_idx], axis=0),
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        **kwargs
                    ).numpy()
                )

            training_batch_loss = np.array(training_batch_loss)
            print(f" Loss:={np.mean(training_batch_loss)}")

            training_loss.append(training_batch_loss)

        training_loss = pd.DataFrame(np.stack(training_loss))
        training_loss.columns.name = "Batch"
        training_loss.index.name = "Epoch"

        return training_loss

if __name__ == "__main__":
    pass
