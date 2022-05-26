import numpy as np

from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import BaseTextEncoder
from .encoder_reduction.base import BaseEncoderReduction
from .encoder_reduction.numpy import NumpyReduction

class PyTorchFinbert (BaseTextEncoder):
    def __init__(self, encoder_reduction: BaseEncoderReduction = NumpyReduction(np.sum)):
        super().__init__(encoder_reduction)

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def flat_encode(self, flat_text_tensor: np.ndarray, **kwargs) -> np.ndarray:
        inputs = self.tokenizer(
            flat_text_tensor,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.model(**inputs)

        # prediction probability distributed: positive (+1) / negative (-1) / neutral (0)
        predictions = softmax(outputs.logits, dim=-1).detach().numpy()

        return predictions[:, 0] - predictions[:, 1]

if __name__ == "__main__":
    pass
