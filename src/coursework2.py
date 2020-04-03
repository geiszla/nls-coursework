"""Main Program for Coursework 2
This module should be run when evaluating coursework 2

To retrain and evaluate the classifier, run `python src/train.py`
"""

from itertools import chain
import re
from typing import Dict

from corpus import Corpus
from utilities import load_sentiment_lexicon

def __run_coursework():
    # PART 1
    # Load the inagural texts
    inagural_corpus = Corpus('Inaugural Texts', 'data/inaugural')

    # Tag with both NLTK's default and the Stanford named entity tagger
    default_tagged = inagural_corpus.get_named_entities_default()
    stanford_tagged = inagural_corpus.get_named_entities_stanford()

    print(f'Default tagger found {len(list(chain.from_iterable(default_tagged)))} named entities')
    # Print named entities of the first text
    print(default_tagged[0])

    stanford_count = len(list(chain.from_iterable(stanford_tagged)))
    print(f'\nStanford tagger found {stanford_count} named entities')
    print(stanford_tagged[0])

    # PART 2
    # Load corpora for sentiment analysis
    positive_review_corpus = Corpus(
        'Positive Movie Reviews',
        'data/rt-polaritydata/rt-polarity.pos',
    )
    negative_review_corpus = Corpus(
        'Negative Movie Reviews',
        ['data/rt-polaritydata/rt-polarity.neg'],
    )

    # PART 2a
    # Merge the negative and positive texts to use them to build a lexicon
    review_corpus = Corpus(
        'Movie Reviews',
        texts=[positive_review_corpus.texts, negative_review_corpus.texts]
    )

    # Load and parse sentiment lexicon seed words
    seed: Dict[str, str] = {}
    with open('data/seed_lexicon.txt', 'r') as seed_reader:
        for line in seed_reader:
            line = line.strip()
            line = line if len(line) == 0 or line[-1] != ';' else line[:-1]

            if line.startswith('Positive adjectives:'):
                for word in re.split(r'Positive adjectives: |,\s*', line.strip())[1:]:
                    seed[word] = 'positive'
            elif line.startswith('Negative adjectives:'):
                for word in re.split(r'Negative adjectives: |,\s*', line.strip())[1:]:
                    seed[word] = 'negative'

    built_lexicon = review_corpus.build_sentiment_lexicon(seed)
    print(built_lexicon)

    # PART 2b
    print('Calculating baseline sentiment metrics...')

    # Load MPQA sentiment lexicon
    sentiment_lexicon = load_sentiment_lexicon(
        'data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    )

    # Use it to evaluate the accuracy of the baseline sentiment classifier
    baseline_accuracy = review_corpus.get_baseline_sentiment_metrics(sentiment_lexicon)

    print(f'Baseline accuracy: {baseline_accuracy}')


if __name__ == '__main__':
    __run_coursework()
