from itertools import chain
import random
import time
from typing import List, Tuple, cast

from classifier import Classifier
from corpus import Corpus
from sklearn.model_selection import KFold
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
        tagger=positive_review_corpus.get_tagger(),
        stemmer=positive_review_corpus.get_stemmer(),
    )

    positive_texts = positive_review_corpus.get_tagged_texts()
    negative_texts = negative_review_corpus.get_tagged_texts()

    vocabulary = positive_review_corpus.get_vocabulary().union(
        negative_review_corpus.get_vocabulary()
    )
    classifier = Classifier(vocabulary, 100)

    print('\nPreprocessing data...')
    processed_data, processed_labels = classifier.preprocess_data(
        list(chain.from_iterable(positive_texts + negative_texts)),
        [0] * len(positive_texts[0]) + [1] * len(negative_texts[0]),
    )

    shuffled = list(zip(processed_data, processed_labels))
    random.shuffle(shuffled)
    data, labels = cast(
        Tuple[List[Tensor], List[Tensor]],
        zip(*shuffled),
    )

    kfold = KFold(5, shuffle=True)
    cumulative_accuracy = 0.0

    print('\nTraining...')
    for training_indices, validation_indices in cast(
        List[Tuple[List[int], List[int]]],
        kfold.split(data, labels),
    ):
        classifier = Classifier(vocabulary, 100)
        optimizer = torch.optim.Adam(classifier.model.parameters())
        best_validation_loss = float('inf')

        training_data = [data[index] for index in training_indices]
        training_labels = [labels[index] for index in training_indices]
        validation_data = [data[index] for index in validation_indices]
        validation_labels = [labels[index] for index in validation_indices]

        for epoch in range(5):
            start_time = time.time()

            training_loss, train_accuracy = classifier.train(
                training_data,
                training_labels,
                optimizer
            )

            validation_loss, validation_accuracy = classifier.evaluate(
                validation_data,
                validation_labels
            )

            epoch_mins = time.time() - start_time
            epoch_secs = int(epoch_mins - (int(epoch_mins / 60) * 60))

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(classifier.model.state_dict(), 'sentiment_bow.pt')

                if epoch == 5:
                    cumulative_accuracy += validation_accuracy

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {training_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Val. Loss: {validation_loss:.3f}'
                + f' |  Val. Acc: {validation_accuracy * 100:.2f}%\n')

    print(f'Average accuracy across folds: {cumulative_accuracy / 5}')


if __name__ == '__main__':
    train_classifier()
