"""Main Program for Coursework 1
This module should be run when evaluating coursework 1
"""

from typing import Dict, List

import en_core_web_sm
from nltk.stem.snowball import SnowballStemmer

from corpus import Corpus


def test_clustering_on_corpus(
    corpus: Corpus, target_words: List[str], stemmer: SnowballStemmer
) -> None:
    print('Testing with full words...')
    corpus.test_clustering(target_words, 2)
    corpus.test_clustering(target_words, 4)
    corpus.test_clustering(target_words, 10)

    # Get the lemma of the word
    word_stems = [stemmer.stem(word) for word in target_words]

    print('\nTesting with word stems...')
    corpus.test_clustering(word_stems, 2)
    corpus.test_clustering(word_stems, 4)
    corpus.test_clustering(word_stems, 10)


if __name__ == '__main__':
    # Load model for POS tagging
    print('Loading tagger...')
    TAGGER = en_core_web_sm.load()

    # PART 1
    # Calculate word likelyhood and word transition probabilities
    PROBABILITIES = Corpus(['data/A_inaugural'], 'A').calculate_vb_nn_probabilities(TAGGER)

    probabilities: Dict[str, float] = {}

    # Print results
    for index, (
        word_likelyhood, to_transition_likelihood, from_transition_likelihood
    ) in enumerate(PROBABILITIES):
        tag = "VB" if index == 0 else "NN"

        print(f'\nWord likelyhood: {word_likelyhood}')
        print(f'Transition likelihood to {tag}: {to_transition_likelihood}')
        print(f'Transition likelihood from {tag}: {from_transition_likelihood}')

        disambiguation_probability = word_likelyhood * to_transition_likelihood \
            * from_transition_likelihood
        probabilities[tag] = disambiguation_probability

        print(f'{tag} disambiguation probability: {disambiguation_probability}')

    PROBABLE_TAG = max(probabilities, key=probabilities.get)
    print(f'\nThis word should most likely be tagged "{PROBABLE_TAG}".')

    # PART 2
    # Load target words
    TARGET_WORDS: List[str] = []
    with open('data/target-words.txt') as word_file:
        TARGET_WORDS += map(str.strip, word_file.readlines())

    B_PATH = 'data/B_ntext'
    C_PATH = 'data/C_hw1-data'

    # Create english stemmer
    STEMMER = SnowballStemmer(language='english')

    # Test clusters for corpus B, C and their combination
    test_clustering_on_corpus(Corpus([B_PATH], 'B'), TARGET_WORDS, STEMMER)
    test_clustering_on_corpus(Corpus([C_PATH], 'C'), TARGET_WORDS, STEMMER)
    test_clustering_on_corpus(Corpus([B_PATH, C_PATH], 'B and C'), TARGET_WORDS, STEMMER)

    print('Exiting...')
