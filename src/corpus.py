"""Corpus for NLP applications
Contains the Corpus class, which represents a set of texts on which NLP operations can be performed.
"""

from collections import Counter
from itertools import chain
from random import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import en_core_web_sm
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import StanfordNERTagger
from sklearn.metrics import accuracy_score
from spacy.language import Language
from tqdm import tqdm

from typings import SentimentLexicon
from utilities import classify_sentiment, create_clusters, get_neighouring_token_count, load_texts

class Corpus():
    """Class, that represents a set of texts on which NLP operations can be performed"""

    def __init__(
        self,
        description: str,
        data_paths: Optional[Union[List[str], str]] = None,
        texts: Optional[List[List[List[str]]]] = None,
        tagger: Optional[Language] = None,
        stemmer: Optional[SnowballStemmer] = None,
    ):
        """Creates a `Corpus` instance

        Parameters
        ----------
        description (str): A few-word description of the current corpus

        data_paths (List[str], Optional): Path or list of paths to look for text files in
            (recursively). Also accepts the text file path directly.

        texts (List[List[List[List[str]]]], Optional): List of texts to add to the corpus

        tagger (Language, Optional): A Spacy tagger used to tokenize and tag texts.
            If none is given, it will be created when needed.

        stemmer (SnowballStemmer, Optional): An NLTK SnowballStemmer used to get stems of the words.
            If none is given, it will be created when needed.
        """

        print(f'\n===== Corpus: {description} =====')

        if data_paths and texts:
            print('Both text paths and texts are specified. Merging them.')

        all_texts: List[List[str]] = []
        if data_paths:
            print('Loading texts...')

            # Load texts from the data path
            if isinstance(data_paths, str):
                all_texts = load_texts(data_paths)
            else:
                all_texts = list(chain.from_iterable(
                    load_texts(data_path) for data_path in data_paths
                ))

        # Load texts from given corpora
        if texts:
            all_texts += chain.from_iterable(texts)

        # Assign initial values to class member variables
        self.texts: List[List[str]] = all_texts
        self.tagged_texts: Union[List[Any], None] = None

        self.tagger: Union[Language, None] = tagger
        self.stemmer: Union[SnowballStemmer, None] = stemmer

    def get_tagger(self) -> Language:
        """Gets the tagger instance if exists, or creates a new one if not

        Returns
        -------
        Language: The tagger instance used to tokenize and tag texts
        """

        if self.tagger:
            return self.tagger

        print('Loading tagger...')
        self.tagger = en_core_web_sm.load()

        return cast(Language, self.tagger)

    def get_stemmer(self) -> SnowballStemmer:
        """Gets the stemmer instance if exists, or creates a new one if not

        Returns
        -------
        SnowballStemmer: The stemmer instance used to get stems of words
        """

        if self.stemmer:
            return self.stemmer

        print('Loading stemmer...')
        self.stemmer = SnowballStemmer(language='english')

        return self.stemmer

    def get_tagged_texts(self) -> List[List[Any]]:
        """Tokenizes and tag all texts

        Returns
        -------
        List[List[Doc]]: List of texts, which consist of Spacy Docs representing lines in the text
        """

        if self.tagged_texts:
            return self.tagged_texts

        print('Tokenizing and tagging texts...')

        tagger = self.get_tagger()
        tagged_texts: List[List[Any]] = []

        # Tag each line of text separately for all texts
        for text in self.texts:
            tagged_texts.append([tagger(line) for line in cast(Any, tqdm(text))])

        self.tagged_texts = tagged_texts
        return self.tagged_texts

    def get_vocabulary(self) -> Set[str]:
        """Generates a set of all the unique words in the corpus texts

        Returns
        -------
        Set[str]: Vocabulary of the corpus texts
        """

        tagged_texts = self.get_tagged_texts()
        return set(word.text for word in chain.from_iterable(chain.from_iterable(tagged_texts)))

    def get_named_entities_default(self) -> List[List[Tuple[str, str]]]:
        """Extracts the named entities from the corpus texts using NLTK's default NER tagger

        Returns
        -------
        List[List[Tuple[str, str]]]: List of texts, which consist of all the named entities in the
            text
        """

        # Download necessary NLTK models
        print('\nDownloading NLTK resources...')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

        print('\nExtracting named entities using the default tagger...')
        entities: List[List[Any]] = []
        for text in cast(Any, tqdm(self.texts)):
            # Join lines of the current text and tag them
            tagged_words = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
            chunks: Any = nltk.ne_chunk(tagged_words)

            # Get the named entities from the tagged entities
            text_entities = []
            for chunk in chunks:
                if type(chunk) is nltk.Tree:
                    name = ''.join(leaf[0] for leaf in chunk.leaves())
                    text_entities.append((name, chunk.label()))

            entities.append(text_entities)

        return entities

    def get_named_entities_stanford(self) -> List[List[Tuple[str, str]]]:
        """Extracts the named entities from the corpus texts using the Stanford NER tagger

        Returns
        -------
        List[List[Tuple[str, str]]]: List of texts, which consist of all the named entities in the
            text
        """

        tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

        print('\nExtracting named entities using the Stanford tagger...')
        entities = []
        for text in cast(Any, tqdm(self.texts)):
            # Join and tag each text
            entities.append(tagger.tag(' '.join(text)))

        return entities

    def calculate_vb_nn_probabilities(
        self
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculates the probabilities that an occurrence of the word "race" has the tag "VB"
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
        context_size: int
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

        # Add the reverse of the target words to the target word list
        test_target_words = target_words + [word[::-1] for word in target_words]
        # Save word indices for faster lookup
        word_to_index = {word: index for index, word in enumerate(test_target_words)}

        stemmer = self.get_stemmer()

        # Create clusters from the modified target words and text
        clusters = create_clusters(test_target_words, cluster_count, context_size, lines, stemmer)
        correct_count = sum(1 for index, cluster in enumerate(clusters[:len(clusters) // 2])
            if cluster == clusters[word_to_index[test_target_words[index][::-1]]])

        print(f'Correct pairs: \033[94m{correct_count}/{cluster_count}\033[0m'
            + f', average accuracy:  \033[94m{correct_count / cluster_count * 100}%\033[0m')

    def build_sentiment_lexicon(self, seed: Dict[str, str]) -> Dict[str, str]:
        """Uses the initial seed of words, their polarity and information obtained from their
            occurrences in the text to add more subjective words to it to build a sentiment
            lexicon.

        Parameters
        ----------
        seed (Dict[str, str]): Initial words the lexicon contains in the format:
            { word: 'positive' or 'negative' }

        Returns
        -------
        Dict[str, str]: The sentiment lexicon extended with additional subjective words
        """

        tagged_texts = self.get_tagged_texts()

        found_words: List[Tuple[str, str]] = []
        for text in tagged_texts:
            for line in text:
                # Get the text of the word from all the tokens in current line
                words = [word.text for word in line]

                for index, word in enumerate(words):
                    # Get new subjective words from the neighbors of the seed words
                    if word in seed and index + 2 < len(words):
                        if words[index + 1] == 'and':
                            found_words.append((words[index + 2], seed[word]))
                        if words[index + 1] == 'but':
                            found_words.append((words[index + 2], (
                                'positive' if seed[word] == 'negative' else 'negative'
                            )))

        # Add all the found words to the dictionary once
        # The words and their polarity are sorted by the number of occurrences, so if a word appears
        # as both 'positive' and 'negative', the one which appears more frequenty will be added to
        # the dictionary
        dictionary: Dict[str, str] = {}
        for word_tuple in dict(Counter(found_words)):
            if word_tuple[0] not in dictionary:
                dictionary[word_tuple[0]] = word_tuple[1]

        return dictionary

    def get_baseline_sentiment_metrics(self, sentiment_lexicon: SentimentLexicon) -> float:
        """Get the accuracy of a basic text sentiment classifier model. This model uses the number
            of positive and negative words to decide the polarity of the text.

        Parameters
        ----------
        sentiment_lexicon (SentimentLexicon): Lexicon to look up the words' polarity in

        Returns
        -------
        float: The accuracy of the model over the corpus texts
        """

        tagger = self.get_tagger()
        stemmer = self.get_stemmer()

        # Assuming first text contains positive and second the negative samples
        labels = [1] * len(self.texts[0]) + [0] * len(self.texts[1])
        # Classify the sentiment of each line in the texts using the given sentiment lexicon
        baseline_predicitons = [classify_sentiment(line, sentiment_lexicon, tagger, stemmer)
            for line in chain.from_iterable(self.texts)]

        # Get the accuracy of the sentiment classification
        return cast(float, accuracy_score(labels, baseline_predicitons))
