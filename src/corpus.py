from functools import reduce
from typing import cast, List, Tuple

import en_core_web_sm
from spacy.language import Language

from utilities import get_neighouring_token_count, load_texts


class Corpus():
    def __init__(self, data_path: str):
        self.tagger: Language = en_core_web_sm.load()

        self.texts: List[Language] = reduce(
            lambda previous_text, current_text:
                cast(List[Language], previous_text.append(self.tagger(' '.join(current_text)))),
            load_texts(data_path),
            [],
        )

    def calculate_vb_nn_probabilities(
        self
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

        # Get word transition counts
        for text in self.texts:
            dt_count += len([token for token in text if token.tag_ == 'DT'])

            dt_vb_count += get_neighouring_token_count('DT', 'VB', text)
            dt_nn_count += get_neighouring_token_count('DT', 'NN', text)

            vb_in_count += get_neighouring_token_count('VB', 'IN', text)
            nn_in_count += get_neighouring_token_count('NN', 'IN', text)

            vb_tokens += [token for token in text if token.tag_ == 'VB']
            nn_tokens += [token for token in text if token.tag_ == 'NN']

        # Calculate probabilities for VB tag
        vb_race_count = len([token for token in vb_tokens if token.text == 'race'])
        vb_word_likelihood = vb_race_count / len(vb_tokens)

        dt_vb_probability = dt_vb_count / dt_count
        vb_in_probability = vb_in_count / len(vb_tokens)

        # Calculate probabilities for NN tag
        nn_race_count = len([token for token in nn_tokens if token.text == 'race'])
        nn_word_likelihood = nn_race_count / len(nn_tokens)

        dt_nn_probability = dt_nn_count / dt_count
        nn_in_probability = nn_in_count / len(nn_tokens)

        return (vb_word_likelihood, dt_vb_probability, vb_in_probability), \
            (nn_word_likelihood, dt_nn_probability, nn_in_probability)
