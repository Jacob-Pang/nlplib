import os
import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from .namespace import get_bert_encoder, get_bert_preprocessor
from ..tensorflow_base import BaseTensorflowTextEncoder
from ..encoder_reduction.tensorflow import TensorflowReduction

class TensorflowBert (BaseTensorflowTextEncoder):
    def __init__(self, bert_handle_name: str, encoder_reduction: TensorflowReduction = TensorflowReduction(tf.reduce_sum),
        dtype: type = tf.float32, **kwargs):

        super(TensorflowBert, self).__init__(encoder_reduction, dtype)
        self.bert_model = self.build_bert_model(bert_handle_name)

    def build_bert_model(self, bert_handle_name: str):
        preprocessing_layer = get_bert_preprocessor(bert_handle_name, name="preprocessing")
        encoder_layer = get_bert_encoder(bert_handle_name, trainable=True, name="encoder")

        text_input = tf_layers.Input(shape=(), dtype=tf.string, name="text")
        encoder_inputs = preprocessing_layer(text_input)
        encoder_outputs = encoder_layer(encoder_inputs)
        pooled_encoder_outputs = encoder_outputs["pooled_output"]
        custom_outputs = self.stack_custom_layers(pooled_encoder_outputs)

        score_output = tf_layers.Dense(
            1, activation=None, name="classifier",
            dtype=self.dtype
        )(custom_outputs)

        return tf.keras.Model(text_input, score_output)

    def stack_custom_layers(self, pooled_encoder_outputs: tf_layers.Layer) -> tf_layers.Layer:
        # override this method to set custom layers
        return tf_layers.Dropout(0.1)(pooled_encoder_outputs)

    def trainable_variables(self) -> list:
        return self.bert_model.trainable_variables

    def flat_encode(self, flat_text_tensor: tf.Tensor) -> tf.Tensor:
        return self.bert_model(flat_text_tensor)

    # Saving and restoring tensorflow models: overriding Pickable methods
    def get_bert_fpath(self, fpath: str) -> str:
        fname, _ = os.path.splitext(fpath)

        return fname + "_analyzer"

    def save_unpickable_attrs(self, fpath: str) -> None:
        self.bert_model.save(self.get_bert_fpath(fpath))

    def restore_unpickable_attrs(self, fpath: str) -> None:
        self.bert_model = tf.keras.models.load_model(self.get_bert_fpath(fpath))

if __name__ == "__main__":
    pass
