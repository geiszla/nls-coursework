"""Utilities
This module contains utility functions to be used by other python scripts and classes.
"""

import os
from typing import List

from spacy.language import Language


def load_texts(data_path: str):
    texts: List[List[str]] = []

    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name), 'r') as input_file:
            # Strip lines and remove empy ones
            lines = [line.strip() for line in input_file.readlines() if len(line) > 0]

            if len(lines) > 0:
                texts.append(lines)

    return texts


def get_neighouring_token_count(first_tag: str, second_tag: str, text: List[Language]) -> int:
    """Gets the number of pair of words, which have the first_tag and second_tag for the first
        word and second word respectively

    Args:
        first_tag (str): Tag of the first word in the pair of words
        second_tag (str): Tag of the second word in the pair of words
        text (List[Language]): Text to search for pairs in

    Returns:
        int: Number of word pairs found
    """

    neighbouring_tokens = [token for index, token in enumerate(text)
        if token.tag_ == first_tag and index + 1 < len(text) and text[index + 1].tag_ == second_tag]

    return len(neighbouring_tokens)
