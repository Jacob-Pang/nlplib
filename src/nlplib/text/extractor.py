from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder

from .utility import collapse_text_sequences

map_n_gram = {
    2: (BigramAssocMeasures,   BigramCollocationFinder),
    3: (TrigramAssocMeasures,  TrigramCollocationFinder),
    4: (QuadgramAssocMeasures, QuadgramCollocationFinder)
}

def extract_n_grams_by_pmi(text: str, n: int, extract_m: int, min_freq: int = 3):
    # supports up to quad-grams
    assert n in map_n_gram
    measures_type, collocation_finder_type = map_n_gram[n]
    
    measures = measures_type()
    collocation_finder = collocation_finder_type.from_words(
        collapse_text_sequences(text)
    )

    collocation_finder.apply_freq_filter(min_freq)
    return collocation_finder.nbest(measures.pmi, extract_m)

if __name__ == "__main__":
    pass
