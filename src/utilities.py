"""Utilities
This module contains utility functions to be used by other python scripts and classes.
"""

from itertools import chain
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Tuple, Union, cast

from nltk.stem.snowball import SnowballStemmer
from nptyping import Array
import numpy
from sklearn.cluster import KMeans
from spacy.language import Language
from torch import Tensor

from typings import SentimentEntry, SentimentLexicon, SplitData

# Public functions

def load_texts(data_path: str) -> List[List[str]]:
    """Loads contents of text files (with any extension) recursively from the given path.

    Parameters
    ----------
    data_path (str): Root path, where the search for files is initiated

    Returns
    -------
    List[List[str]]: List of text (list of list of lines) read in from each text file.
    """

    texts: List[List[str]] = []

    if (os.path.isfile(data_path)):
        texts.append(__load_text(data_path))

    # Get all files recursively in the given directory
    for file_name in (path for path in Path(data_path).rglob('*') if path.is_file()):
        lines = __load_text(file_name)

        # Add text if it has at least one line
        if len(lines) > 0:
            texts.append(lines)

    return texts


def __load_text(file_name: Union[str, Path]) -> List[str]:
    lines = []

    with open(file_name, 'r') as input_file:
        # Strip lines and remove empy ones
        lines = [line for line in map(str.strip, input_file.readlines()) if len(line) > 0]

    return lines


def get_neighouring_token_count(first_tag: str, second_tag: str, text: List[Language]) -> int:
    """Gets the number of pair of words, which have the first_tag and second_tag for the first
        word and second word respectively.

    Args
    ----
    first_tag (str): Tag of the first word in the pair of words

    second_tag (str): Tag of the second word in the pair of words

    text (List[Language]): Text to search for pairs in

    Returns
    -------
    int: Number of word pairs found
    """

    # For all token with the correct tag, get the following token, examine its tag and count them
    # if they match the given tags
    return sum(1 for index, token in enumerate(text)
        if cast(Any, token).tag_ == first_tag and index + 1 < len(text)
            and cast(Any, text[index + 1]).tag_ == second_tag)


def get_word_context(
    word_index: int,
    words: List[str],
    context_size: int,
    stemmer: Union[SnowballStemmer, None] = None
) -> List[str]:
    """Get context of a word in a word list.

    Parameters
    ----------
    word_index (int): Index of the target word in the word list, of which the context is obtained

    words (List[str]): Word list to look for `word_index` and the context in.

    context_size (int): Number of words to get surrounding the target word (`word_index`)

    stemmer (SnowballStemmer, Optional): The stemmer to use to get stems. If not passed, the whole
        word will be used.

    Returns
    -------
    List[str]: List of words surrounding the target word. Same number of words from before and after
        the target word (when possible).
    """

    # Get index of first element in context
    start_index = word_index - context_size // 2
    start_index = start_index if start_index >= 0 else 0

    # Get index of last element in context
    end_index = word_index + context_size - (word_index - start_index)
    end_index = end_index if end_index < len(words) else len(words) - 1

    # Calculate current context size
    context = [words[index] for index in range(start_index, end_index + 1) if index != word_index]
    current_size = len(context)

    # If we are at the end of the list and the size of the context is less than what's given
    if current_size <= context_size:
        # Add more elements at the start if possible
        start_index = start_index - (context_size - current_size)
        start_index = start_index if start_index >= 0 else 0

    # Get items from start to end index not including the target word itself
    context = [words[index] for index in range(start_index, end_index + 1) if index != word_index]

    # If stemmer is passed, use it to get words' stems
    return [cast(Any, stemmer).stem(word) for word in context] if stemmer else context


def create_clusters(
    target_words: List[str],
    cluster_count: int,
    context_size: int,
    lines: List[List[str]],
    stemmer: Union[SnowballStemmer, None] = None
) -> Array[numpy.int32, None, None]:  # type: ignore
    """Creates clusters from given words using their co-occurrence count as a metric.

    Args
    ----
    target_words (List[str]): List of words to cluster

    cluster_count (int): Number of clusters to create

    context_size (int): Overall number of words to examine around the target word when
        counting co-occurrences

    lines (List[List[str]]): Lines of words in which the co-occurrence patterns are searched

    Returns
    -------
    Array[numpy.int32, None, None]: 2D array of integers representing the number of times two
        words co-occur
    """

    # Get all unique words (vocabulary)
    vocabulary = list(set(chain.from_iterable(lines)))

    # If stemmer is passed, use it to get stems of words in vocabulary
    if stemmer:
        vocabulary = [cast(Any, stemmer).stem(word) for word in vocabulary]

    # Store word indexes for faster lookups
    target_word_to_index = {word: index for index, word in enumerate(target_words)}
    vocabulary_word_to_index = {word: index for index, word in enumerate(vocabulary)}

    print('\nBuilding co-occurrence array...')

    # Initiate a 2D numpy array for the word x word matrix filled with zeroes
    word_x_word_matrix = numpy.zeros((len(target_words), len(vocabulary)), numpy.int32)

    # For all words in all lines
    for line in lines:
        for index, word in enumerate(line):
            # Get context of current word
            context = get_word_context(index, line, context_size, stemmer)

            # If the word is among the target words
            if word in target_words:
                context_indexes = [vocabulary_word_to_index[context_word]
                    for context_word in context]

                # increment all co-occurrence counts for the context words
                word_x_word_matrix[target_word_to_index[word], context_indexes] += 1

    print(f'Clustering (windows size: {context_size})...')
    return cast(
        Array[numpy.int32, None, None],  # type: ignore
        KMeans(cluster_count).fit_predict(word_x_word_matrix)
    )


def load_sentiment_lexicon(path: str) -> Dict[str, Dict[str, str]]:
    lexicon: Dict[str, Dict[str, str]] = {}

    with open(path, 'r') as lexicon_file:
        for line in lexicon_file:
            word = ''
            dictionary: Dict[str, str] = {}

            for property in re.split(r'\s+', line):
                if property == '' or '=' not in property:
                    continue

                [key, value] = property.split('=')

                if key == 'word1':
                    word = value
                else:
                    dictionary[key] = value

            lexicon[word] = dictionary

    return lexicon

def classify_sentiment(
    text: str,
    lexicon: SentimentLexicon,
    tagger: Language,
    stemmer: SnowballStemmer,
    is_use_stem: bool = True,
    is_ignore_pos: bool = False,
) -> int:
    polarities: Dict[str, Dict[str, SentimentEntry]] = {'positive': {}, 'negative': {}}

    pos_values = {
        'adverb': 'ADV',
        'noun': 'NOUN',
        'verb': 'VERB',
        'adj': 'ADJ',
    }

    for word_token in cast(Any, tagger(text)):
        word = word_token.text
        stemmed_word = cast(Any, stemmer).stem(word)

        if (
            (word in lexicon
                and lexicon[word]['priorpolarity'] in ['positive', 'negative']
                and (not is_ignore_pos and lexicon[word]['pos1'] == 'anypos'
                    or pos_values[lexicon[word]['pos1']] == word_token.pos_))
            or is_use_stem and stemmed_word in lexicon
                and (is_ignore_pos or lexicon[stemmed_word]['stemmed1'] == 'y')
        ):
            polarities[lexicon[word]['priorpolarity']][word] = lexicon[word]

    return 0 if len(polarities['positive']) > len(polarities['negative']) else 1

def split_dataset(
    batches: List[Tensor], labels: List[Tensor], training_fraction: float
) -> SplitData:
    # Shuffle images with their labels
    shuffled = list(zip(batches, labels))
    random.shuffle(shuffled)
    shuffled_batches, shuffled_labels = cast(
        Tuple[List[Tensor], List[Tensor]],
        zip(*shuffled),
    )

    # Separate shuffled dataset to training and validation sets
    sample_count = len(shuffled_batches)
    training_count = int(sample_count * training_fraction)

    return (
        list(shuffled_batches[0:training_count]),
        list(shuffled_labels[0:training_count]),
        list(shuffled_batches[training_count:sample_count]),
        list(shuffled_labels[training_count:sample_count])
    )
