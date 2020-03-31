"""Main Program for Coursework 1
This module should be run when evaluating coursework 1
"""

from typing import Dict, List

import en_core_web_sm
from nltk.stem.snowball import SnowballStemmer
from spacy.language import Language

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


def run_coursework():
    # Load model for POS tagging
    print('Loading tagger...')
    tagger: Language = en_core_web_sm.load()

    # PART 1
    # Calculate word likelyhood and word transition probabilities
    probabilities = Corpus(['data/A_inaugural'], 'A').calculate_vb_nn_probabilities(tagger)
    cumulative_probabilities: Dict[str, float] = {}

    # Print results
    for index, (
        word_likelyhood, to_transition_likelihood, from_transition_likelihood
    ) in enumerate(probabilities):
        tag = "VB" if index == 0 else "NN"

        print(f'\nWord likelyhood: {word_likelyhood}')
        print(f'Transition likelihood to {tag}: {to_transition_likelihood}')
        print(f'Transition likelihood from {tag}: {from_transition_likelihood}')

        disambiguation_probability = word_likelyhood * to_transition_likelihood \
            * from_transition_likelihood
        cumulative_probabilities[tag] = disambiguation_probability

        print(f'{tag} disambiguation probability: {disambiguation_probability}')

    probable_tag = max(cumulative_probabilities, key=cumulative_probabilities.get)
    print(f'\nThis word should most likely be tagged "{probable_tag}".')

    # PART 2
    # Load target words
    target_words: List[str] = []
    with open('data/target-words.txt') as word_file:
        target_words += map(str.strip, word_file.readlines())

    b_path = 'data/B_ntext'
    c_path = 'data/C_hw1-data'

    # Create english stemmer
    stemmer = SnowballStemmer(language='english')

    # Test clusters for corpus B, C and their combination
    test_clustering_on_corpus(Corpus([b_path], 'B'), target_words, stemmer)
    test_clustering_on_corpus(Corpus([c_path], 'C'), target_words, stemmer)
    test_clustering_on_corpus(Corpus([b_path, c_path], 'B and C'), target_words, stemmer)

    print('Exiting...')


if __name__ == '__main__':
    run_coursework()
