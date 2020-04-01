"""Corpus for NLP applications
Contains the Corpus class, which represents a set of texts on which NLP operations can be performed.
"""

from itertools import chain
from random import random
from typing import Any, List, Optional, Set, Tuple, Union, cast

import en_core_web_sm
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import StanfordNERTagger
from sklearn.metrics import precision_recall_fscore_support
from spacy.language import Language
from tqdm import tqdm

from typings import ClassificationMetrics, SentimentLexicon
from utilities import classify_sentiment, create_clusters, get_neighouring_token_count, load_texts

class Corpus():
    """Class, that represents a set of texts on which NLP operations can be performed."""

    def __init__(
        self,
        description: str,
        data_paths: Optional[Union[List[str], str]] = None,
        corpora: Optional[List['Corpus']] = None,
        tagger: Optional[Language] = None,
        stemmer: Optional[SnowballStemmer] = None,
    ):
        """Creates a `Corpus` instance

        Parameters
        ----------
        data_paths (List[str]): List of paths to look for text files in (recursively)

        description (str): A short description of the current corpus
        """

        print(f'\n===== Corpus: {description} =====')

        if data_paths and corpora:
            print('Both text paths and corpora are specified. Merging them.')

        texts: List[List[str]] = []
        if data_paths:
            # Load and join lines in each text and tag the produced texts
            print('Loading texts...')

            if isinstance(data_paths, str):
                texts = load_texts(data_paths)
            else:
                texts = list(chain.from_iterable(load_texts(data_path) for data_path in data_paths))

        if corpora:
            texts += chain.from_iterable(corpus.texts for corpus in corpora)

        self.texts: List[List[str]] = texts
        self.tagger: Union[Language, None] = tagger
        self.stemmer: Union[SnowballStemmer, None] = stemmer

    def get_vocabulary(self) -> Set[str]:
        if not self.tagger:
            print('Error: please pass tagger to the Corpus\' constructor to tag the texts')
            return Set()

        tagged_texts: List[List[Any]] = []

        for text in self.texts:
            tagged_texts.append([self.tagger(line) for line in text])

        return set(word.text for word in chain.from_iterable(tagged_texts))

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

        print('Tagging texts...')
        tagger = self.get_tagger()
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

    def get_named_entities_default(self) -> List[List[Tuple[str, str]]]:
        print('\nDownloading NLTK resources...')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

        print('\nExtracting named entities using the default tagger...')
        entities: List[List[Any]] = []
        for text in cast(Any, tqdm(self.texts)):
            tagged_words = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
            chunks: Any = nltk.ne_chunk(tagged_words)

            text_entities = []
            for chunk in chunks:
                if type(chunk) is nltk.Tree:
                    name = ''.join(leaf[0] for leaf in chunk.leaves())
                    text_entities.append((name, chunk.label()))

            entities.append(text_entities)

        return entities

    def get_named_entities_stanford(self) -> List[List[Tuple[str, str]]]:
        tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

        print('\nExtracting named entities using the Stanford tagger...')
        entities = []
        for text in cast(Any, tqdm(self.texts)):
            entities.append(tagger.tag(' '.join(text)))

        return entities

    def get_baseline_sentiment_metrics(
        self, sentiment_lexicon: SentimentLexicon
    ) -> ClassificationMetrics:
        tagger = self.get_tagger()
        stemmer = self.get_stemmer()

        # Assuming first text contains positive and second the negative samples
        ground_truths = [1] * len(self.texts[0]) + [0] * len(self.texts[1])
        baseline_predicitons = [classify_sentiment(line, sentiment_lexicon, tagger, stemmer)
            for line in chain.from_iterable(self.texts)]

        return cast(
            ClassificationMetrics,
            precision_recall_fscore_support(ground_truths, baseline_predicitons),
        )

    def get_tagger(self) -> Language:
        if self.tagger:
            return self.tagger

        print('Loading tagger...')
        self.tagger = en_core_web_sm.load()

        return cast(Language, self.tagger)

    def get_stemmer(self) -> SnowballStemmer:
        if self.stemmer:
            return self.stemmer

        print('Loading stemmer...')
        self.stemmer = SnowballStemmer(language='english')

        return self.stemmer
