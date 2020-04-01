"""Corpus for NLP applications
Contains the Corpus class, which represents a set of texts on which NLP operations can be performed.
"""

from itertools import chain
from random import random
from typing import Any, List, Tuple, Union, cast

from nltk.stem.snowball import SnowballStemmer
from spacy.language import Language

from utilities import create_clusters, get_neighouring_token_count, load_texts


class Corpus():
    """Class, that represents a set of texts on which NLP operations can be performed."""

    def __init__(self, data_paths: List[str], description: str):
        """Creates a `Corpus` instance

        Parameters
        ----------
        data_paths (List[str]): List of paths to look for text files in (recursively)

        description (str): A short description of the current corpus
        """

        print(f'\n===== Corpus: {description} =====')

        # Load and join lines in each text and tag the produced texts
        print('Loading texts...')
        self.texts = list(chain.from_iterable(load_texts(data_path) for data_path in data_paths))

    def calculate_vb_nn_probabilities(
        self, tagger: Language
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate the probabilities that an occurrence of the word "race" has the tag "VB"
            and the same for "NN".

        Returns
        -------
        Tuple[float, float]: Calculated probabilities for the "VB" and "NN" tags respectively
        """

        dt_count = 0

        dt_vb_count = 0
        dt_nn_count = 0

        vb_in_count = 0
        nn_in_count = 0

        vb_tokens: List[Language] = []
        nn_tokens: List[Language] = []

        print('Tagging texts...')
        tagged_texts: List[Any] = [tagger(' '.join(text)) for text in self.texts]

        # Get word transition counts
        for text in tagged_texts:
            dt_count += len([token for token in text if token.tag_ == 'DT'])

            dt_vb_count += get_neighouring_token_count('DT', 'VB', text)
            dt_nn_count += get_neighouring_token_count('DT', 'NN', text)

            vb_in_count += get_neighouring_token_count('VB', 'IN', text)
            nn_in_count += get_neighouring_token_count('NN', 'IN', text)

            vb_tokens += [token for token in text if token.tag_ == 'VB']
            nn_tokens += [token for token in text if token.tag_ == 'NN']

        # Calculate probabilities for VB tag
        vb_race_count = sum(token for token in vb_tokens if cast(Any, token).text == 'race')
        vb_word_likelihood = vb_race_count / len(vb_tokens)

        dt_vb_probability = dt_vb_count / dt_count
        vb_in_probability = vb_in_count / len(vb_tokens)

        # Calculate probabilities for NN tag
        nn_race_count = len([token for token in nn_tokens if cast(Any, token).text == 'race'])
        nn_word_likelihood = nn_race_count / len(nn_tokens)

        dt_nn_probability = dt_nn_count / dt_count
        nn_in_probability = nn_in_count / len(nn_tokens)

        return (vb_word_likelihood, dt_vb_probability, vb_in_probability), \
            (nn_word_likelihood, dt_nn_probability, nn_in_probability)

    def test_clustering(
        self,
        target_words: List[str],
        context_size: int,
        stemmer: Union[SnowballStemmer, None] = None
    ) -> None:
        """Creates clusters from the given word list and tests its accuracy using pseudoword
            disambiguation on the corpus texts.

        Parameters
        ----------
        target_words (List[str]): List of words to test the clustering algorithm on.

        context_size (int): Number of words to be considered around the target word when building
            a co-occurrence array
        """

        cluster_count = len(target_words)

        # Reverse all words in all lines with 50% probability
        lines = list(map(str.split, chain.from_iterable(self.texts)))
        for index, line in enumerate(lines):
            lines[index] = list(map(lambda word: word[::-1] if random() > 0.5 else word, line))

        # Add the reversed version of the target words to the target word list
        test_target_words = target_words + [word[::-1] for word in target_words]
        # Save word indexes for faster lookup
        word_to_index = {word: index for index, word in enumerate(test_target_words)}

        # Create clusters from the modified target words and text
        clusters = create_clusters(test_target_words, cluster_count, context_size, lines, stemmer)
        correct_count = sum(1 for index, cluster in enumerate(clusters[:len(clusters) // 2])
            if cluster == clusters[word_to_index[test_target_words[index][::-1]]])

        print(f'Correct pairs: \033[94m{correct_count}/{cluster_count}\033[0m'
            + f', average accuracy:  \033[94m{correct_count / cluster_count * 100}%\033[0m')
