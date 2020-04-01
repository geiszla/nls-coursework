"""Main Program for Coursework 2
This module should be run when evaluating coursework 2
"""

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

    review_corpus = Corpus('Movie Reviews', 'data/rt-polaritydata')
    sentiment_lexicon = load_sentiment_lexicon(
        'data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    )

    print('Calculating baseline sentiment metrics...')
    baseline_metrics = review_corpus.get_baseline_sentiment_metrics(sentiment_lexicon)
    print(baseline_metrics)

    classifier = Classifier(review_corpus.get_vocabulary(), 100, 4)
    classifier.load('sentiment_rnn.pt')


if __name__ == '__main__':
    run_coursework()
