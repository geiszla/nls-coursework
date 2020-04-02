"""Main Program for Coursework 2
This module should be run when evaluating coursework 2
"""

from itertools import chain
import re
from typing import Any, Dict, List, cast

import torch

from classifier import Classifier
from corpus import Corpus
from utilities import load_sentiment_lexicon

def run_coursework():
    # PART 1
    inagural_corpus = Corpus('Inaugural Texts', 'data/inaugural')

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
    review_corpus = Corpus(
        'Movie Reviews',
        corpora=[positive_review_corpus, negative_review_corpus]
    )

    # PART 2a
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
    vocabulary = positive_review_corpus.get_vocabulary().union(
        negative_review_corpus.get_vocabulary(),
    )
    classifier = Classifier(vocabulary, 100, 4)

    texts = positive_review_corpus.texts + negative_review_corpus.texts

    data = classifier.preprocess_data(list(chain.from_iterable(texts)))
    labels: List[torch.Tensor] = cast(Any, torch).IntTensor(
        [0] * len(positive_review_corpus.texts) + [1] * len(negative_review_corpus.texts)
    )

    print('Evaluating classifier...')
    mean_accuracy = classifier.test(data, labels)

    print('Calculating baseline sentiment metrics...')

    # Load sentiment lexicon
    sentiment_lexicon = load_sentiment_lexicon(
        'data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    )
    baseline_accuracy = review_corpus.get_baseline_sentiment_metrics(sentiment_lexicon)

    print(f'Mean accuracy: {mean_accuracy} (baseline: {baseline_accuracy})')


if __name__ == '__main__':
    run_coursework()
