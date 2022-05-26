import re
import string
import operator

from functools import reduce
from nltk import tokenize
from nltk.corpus import stopwords

from .utility import apply_on_text_sequences
from ..text import regex_patterns
from .nltk_downloads import *

class drop_token_cond:
    def single_char_only(token: str) -> bool:
        return not (re.match(regex_patterns.single_char_only, token) is None)

    def punctuation(token: str) -> bool:
        return token in string.punctuation

    def stopword(token: str) -> bool:
        return token in stopwords.words("english")

    def empty(token: str) -> bool:
        return token == ''

    def disjoint_drop_token_cond(*drop_token_conds: callable) -> callable:
        return lambda token: reduce(operator.or_, [
            drop_token_cond(token) for drop_token_cond in drop_token_conds
        ])

def drop_tokens_by_cond(token_list: list, drop_token_cond: callable):
    return [
        token for token in token_list
        if not drop_token_cond(token)
    ]

@apply_on_text_sequences
def word_tokenize(text: str, drop_token_cond: callable = drop_token_cond.empty) -> list:
    return drop_tokens_by_cond(tokenize.word_tokenize(text), drop_token_cond)

@apply_on_text_sequences
def window_tokenize(text: str, spans: int, drop_token_cond: callable = drop_token_cond.empty,
    drop_window_token_cond: callable = drop_token_cond.empty, delimiter: str = ' ') -> list:

    if isinstance(spans, int):
        spans = range(spans)

    token_list = word_tokenize(text, drop_token_cond)
    num_tokens = len(token_list)

    window_token_list = reduce(
        operator.add, [
            [
                delimiter.join(token_list[t:t + span])
                for t in range(num_tokens - span + 1)
            ] for span in spans
        ]
    )

    return drop_tokens_by_cond(window_token_list, drop_window_token_cond)


if __name__ == "__main__":
    pass
