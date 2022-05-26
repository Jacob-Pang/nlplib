import numpy as np
import pandas as pd

def polarity_preserving(transformer: callable):
    def wrapped_transformer(target_scores: np.ndarray, min_score: float,
        max_score: float, *args, preserve_polarity: bool = False, **kwargs):

        if not preserve_polarity:
            return transformer(target_scores, min_score, max_score, *args, **kwargs)
        
        target_scores = target_scores.copy()
        target_scores[target_scores > 0] = transformer(target_scores[target_scores > 0],
                0, max_score, *args, **kwargs)
        target_scores[target_scores < 0] = transformer(target_scores[target_scores < 0],
                min_score, 0, *args, **kwargs)

        return target_scores

    return wrapped_transformer

@polarity_preserving
def normalize_scores(target_scores: np.ndarray, min_score: float = 0,
    max_score: float = 1) -> np.ndarray:

    # Returns normalized version of <target_scores> bounded between <min_score>, <max_score>.
    min_target = np.min(target_scores)
    max_target = np.max(target_scores)

    return min_score + (max_score - min_score) * \
        (target_scores - min_target) / (max_target - min_target)

@polarity_preserving
def rolling_normalize_window(target_scores: np.ndarray, min_score: float = 0,
    max_score: float = 1, window: int = None) -> np.ndarray:
    if window is None:
        window = max(target_scores.shape[0] // 3, 1)

    concat = pd.concat if isinstance(target_scores, (pd.Index, pd.Series, pd.DataFrame)) \
            else np.concatenate

    return concat([
        normalize_scores(target_scores[:window], min_score, max_score), *[
            normalize_scores(target_scores[t: t + window], min_score, max_score)[-1:]
            for t in range(1, target_scores.shape[0] - window + 1)
        ]
    ], axis=0)

if __name__ == "__main__":
    pass
