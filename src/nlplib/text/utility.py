import operator

from functools import reduce

def apply_on_text_sequences(method: callable):
    def wrapped_apply(text_sequences, *args, **kwargs):
        if isinstance(text_sequences, str):
            return method(text_sequences, *args, **kwargs)

        return [
            wrapped_apply(text_subsequences, *args, **kwargs)
            for text_subsequences in text_sequences
        ]

    return wrapped_apply

def collapse_text_sequences(text_sequences: list) -> list:
    if isinstance(text_sequences, str):
        return [text_sequences]

    if not text_sequences:
        return []

    return reduce(
        operator.add, [
            collapse_text_sequences(text_subsequences)
            for text_subsequences in text_sequences
        ]
    )

if __name__ == "__main__":
    pass
