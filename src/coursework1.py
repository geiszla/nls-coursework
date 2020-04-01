"""Main Program for Coursework 1
This module should be run when evaluating coursework 1
"""

from typing import Dict, List, cast

from nltk.stem.snowball import SnowballStemmer

from corpus import Corpus


def test_clustering_on_corpus(corpus: Corpus, target_words: List[str]) -> None:
    print('Testing with full words...')
    corpus.test_clustering(target_words, 2)
    corpus.test_clustering(target_words, 4)
    corpus.test_clustering(target_words, 10)

    # Get the lemma of the word
    word_stems = [cast(SnowballStemmer, corpus.stemmer).stem(word) for word in target_words]

    print('\nTesting with word stems...')
    corpus.test_clustering(word_stems, 2)
    corpus.test_clustering(word_stems, 4)
    corpus.test_clustering(word_stems, 10)


def run_coursework():
    # PART 1
    # Calculate word likelyhood and word transition probabilities
    probabilities = Corpus('A', 'data/inaugural').calculate_vb_nn_probabilities()
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

    # Create english stemmer
    b_corpus = Corpus('B', 'data/ntext')
    tagger = b_corpus.get_tagger()
    stemmer = b_corpus.get_stemmer()

    c_corpus = Corpus('C', 'data/hw1-data', tagger=tagger, stemmer=stemmer)
    b_c_corpus = Corpus('B and C', corpora=[b_corpus, c_corpus], tagger=tagger, stemmer=stemmer)

    # Test clusters for corpus B, C and their combination
    test_clustering_on_corpus(b_corpus, target_words)
    test_clustering_on_corpus(c_corpus, target_words)
    test_clustering_on_corpus(b_c_corpus, target_words)

    print('Exiting...')


if __name__ == '__main__':
    run_coursework()
