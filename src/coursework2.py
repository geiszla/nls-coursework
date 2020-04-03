"""Main Program for Coursework 2
This module should be run when evaluating coursework 2

To retrain and evaluate the classifier, run `python src/train.py`
"""

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

    print(default_tagged)
    print(stanford_tagged)

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
            if line.startsWith('Positive adjectives:'):
                for word in re.split(r'(Positiv adjectives: |,\s*)', line)[1:]:
                    seed[word] = 'positive'
            elif line.startsWith('Negative adjectives:'):
                for word in re.split(r'(Negative adjectives: |,\s*)', line)[1:]:
                    seed[word] = 'negative'

    print(review_corpus.build_sentiment_lexicon(seed))

    # PART 2b
    print('Calculating baseline sentiment metrics...')

    # Load MPQA sentiment lexicon
    sentiment_lexicon = load_sentiment_lexicon(
        'data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    )

    # Use it to evaluate the accuracy of the baseline sentiment classifier
    baseline_accuracy = review_corpus.get_baseline_sentiment_metrics(sentiment_lexicon)

    print(f'Baseline accuracy: {baseline_accuracy})')


if __name__ == '__main__':
    __run_coursework()
