"""Main Program for Coursework 1
This module should be run when evaluating coursework 1
"""

from typing import List

import en_core_web_sm

from corpus import Corpus


if __name__ == '__main__':
    # Load model for POS tagging
    print('Loading tagger...')
    TAGGER = en_core_web_sm.load()

    # PART 1
    # # Calculate word likelyhood and word transition probabilities
    # PROBABILITIES = Corpus(['data/A_inaugural'], TAGGER).calculate_vb_nn_probabilities()

    # # Print results
    # for index, (
    #     word_likelyhood, to_transition_likelihood, from_transition_likelihood
    # ) in enumerate(PROBABILITIES):
    #     tag = "VB" if index == 1 else "NN"

    #     print(f'\nWord likelyhood: {word_likelyhood}')
    #     print(f'Transition likelihood to {tag}: {to_transition_likelihood}')
    #     print(f'Transition likelihood from {tag}: {from_transition_likelihood}')

    # PART 2
    # Load target words
    TARGET_WORDS: List[str] = []
    with open('data/target-words.txt') as word_file:
        TARGET_WORDS += map(str.strip, word_file.readlines())

    B_PATH = 'data/B_ntext'
    C_PATH = 'data/C_hw1-data'

    # Create clusters from corpus B
    B_CORPUS = Corpus([B_PATH], TAGGER, 'B')
    B_CORPUS.test_clustering(TARGET_WORDS, 4)

    # Create clusters from corpus C
    C_CORPUS = Corpus([C_PATH], TAGGER, 'C')
    C_CORPUS.test_clustering(TARGET_WORDS, 4)

    # Create clusters from corpus B and C concatenated
    BC_CORPUS = Corpus([B_PATH, C_PATH], TAGGER, 'B and C')
    BC_CORPUS.test_clustering(TARGET_WORDS, 4)
