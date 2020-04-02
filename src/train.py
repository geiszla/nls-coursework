from itertools import chain
import time
from typing import Any, List, Tuple, cast

import numpy
from sklearn.model_selection import KFold

from classifier import Classifier
from corpus import Corpus
import torch
from torch import Tensor

def train_classifier():
    positive_review_corpus = Corpus(
        'Positive Movie Reviews',
        'data/rt-polaritydata/rt-polarity.pos',
    )
    negative_review_corpus = Corpus(
        'Negative Movie Reviews',
        ['data/rt-polaritydata/rt-polarity.neg'],
    )

    vocabulary = positive_review_corpus.get_vocabulary().union(
        negative_review_corpus.get_vocabulary(),
    )
    classifier = Classifier(vocabulary, 100, 4)

    positive_count = len(positive_review_corpus.texts)
    negative_count = len(negative_review_corpus.texts)

    texts = positive_review_corpus.texts + negative_review_corpus.texts

    data = classifier.preprocess_data(list(chain.from_iterable(texts)))
    labels: List[Tensor] = cast(Any, torch).IntTensor([0] * positive_count + [1] * negative_count)

    kfold = KFold(n_splits=10)

    for train_indices, test_indices in cast(List[Tuple[List[int], List[int]]], kfold.split(data)):
        training_data = numpy.array(data)[train_indices]
        training_labels = numpy.array(labels)[train_indices]
        evaluation_data = numpy.array(data)[test_indices]
        evaluation_labels = numpy.array(labels)[test_indices]

        best_valid_loss = float('inf')

        for epoch in range(50):
            start_time = time.time()

            train_loss, train_acc = classifier.train(training_data, training_labels)
            valid_loss, valid_acc = classifier.evaluate(evaluation_data, evaluation_labels)

            epoch_mins = time.time() - start_time
            epoch_secs = int(epoch_mins - (int(epoch_mins / 60) * 60))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(classifier.model.state_dict(), 'sentiment_rnn.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


if __name__ == '__main__':
    train_classifier()
