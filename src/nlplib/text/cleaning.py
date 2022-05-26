import re

from .utility import apply_on_text_sequences    

def drop_empty_sequences(sequences):
    # Recursively removes subsequences that are empty or None
    if sequences is None or isinstance(sequences, str):
        return sequences

    _sequences = [
        drop_empty_sequences(subsequences)
        for subsequences in sequences
    ]

    return [
        subsequences for subsequences in _sequences
        if subsequences
    ]

@apply_on_text_sequences
def to_lowercase(text: str) -> str:
    return text.lower()

@apply_on_text_sequences
def remove_regex(text: str, regex: str) -> str:
    return re.sub(regex, '', text)

@apply_on_text_sequences
def remove_double_spaces(text: str) -> str:
    return ' '.join(text.split())

if __name__ == "__main__":
    pass
