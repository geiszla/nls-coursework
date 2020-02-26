"""Main Program for Coursework 1
This module should be run when evaluating coursework 1
"""

from typing import List

from corpus import Corpus
from embeddings import Embeddings
from utilities import load_texts


if __name__ == '__main__':
    # PART 1
    # Calculate word likelyhood and word transition probabilities
    for index, (
        word_likelyhood, to_transition_likelihood, from_transition_likelihood
    ) in enumerate(Corpus('data/A_inaugural').calculate_vb_nn_probabilities()):
        tag = "VB" if index == 1 else "NN"

        print(f'Word likelyhood: {word_likelyhood}')
        print(f'Transition likelihood to {tag}: {to_transition_likelihood}')
        print(f'Transition likelihood from {tag}: {from_transition_likelihood}')

    # PART 2
    target_words: List[str] = []
    with open('data/target-words.txt') as word_file:
        target_words += word_file.readlines()

    EMBEDDINGS = Embeddings(target_words, 16, 4)
    EMBEDDINGS.train(load_texts('data/B_ntext'))
    EMBEDDINGS.test(len(target_words))
