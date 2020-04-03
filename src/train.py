from itertools import chain
import random
import time
from typing import List, Tuple, cast

from classifier import Classifier
from corpus import Corpus
from sklearn.model_selection import KFold
import torch
from torch import Tensor

def __train_classifier():
    # Load polarity corpora
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

    # Tokenize corpora to get the words in them
    print('')
    positive_texts = positive_review_corpus.get_tagged_texts()
    negative_texts = negative_review_corpus.get_tagged_texts()

    # Generate the vocabulary of all texts and create a classifier from it
    print('Generating vocabulary...')
    vocabulary = positive_review_corpus.get_vocabulary().union(
        negative_review_corpus.get_vocabulary()
    )
    classifier = Classifier(vocabulary, 100)

    # Preprocess texts to be in the correct input format for the classifier model
    # (i.e. create input data and label tensors)
    print('\nPreprocessing data...')
    processed_data, processed_labels = classifier.preprocess_data(
        list(chain.from_iterable(positive_texts + negative_texts)),
        [0] * len(positive_texts[0]) + [1] * len(negative_texts[0]),
    )

    # Shuffle texts, so that the positive and negative samples are not together
    shuffled = list(zip(processed_data, processed_labels))
    random.shuffle(shuffled)
    data, labels = cast(
        Tuple[List[Tensor], List[Tensor]],
        zip(*shuffled),
    )

    # Create k-fold splits
    fold_count = 5
    kfold = KFold(fold_count, shuffle=True)
    cumulative_accuracy = 0.0
    best_validation_loss = float('inf')

    print('\nTraining classifier...')
    for fold, (training_indices, validation_indices) in enumerate(cast(
        List[Tuple[List[int], List[int]]],
        kfold.split(data, labels),
    )):
        print(f'\n===== Fold: {fold + 1}/{fold_count} =====')

        # For each fold, get the training and validation data
        training_data = [data[index] for index in training_indices]
        training_labels = [labels[index] for index in training_indices]
        validation_data = [data[index] for index in validation_indices]
        validation_labels = [labels[index] for index in validation_indices]

        # Create a new classifier (so it won't be evaluated on samples that it's already trained on)
        classifier = Classifier(vocabulary, 100)
        optimizer = torch.optim.Adam(classifier.model.parameters())

        # Train for 5 epochs
        for epoch in range(5):
            start_time = time.time()

            # Train it on all training texts
            training_loss, train_accuracy = classifier.train(
                training_data,
                training_labels,
                optimizer
            )

            # Validate training on validation texts
            validation_loss, validation_accuracy = classifier.evaluate(
                validation_data,
                validation_labels
            )

            epoch_mins = time.time() - start_time
            epoch_secs = int(epoch_mins - (int(epoch_mins / 60) * 60))

            # If the validation loss is less than the best, update it and save the model
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(classifier.model.state_dict(), 'sentiment_bow.pt')

            # Print the statistics for the current epoch
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {training_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Val. Loss: {validation_loss:.3f}'
                + f' |  Val. Acc: {validation_accuracy * 100:.2f}%\n')

            # Add the accuracy to the cumulative accuracy at the end of each fold
            if epoch == 4:
                cumulative_accuracy += validation_accuracy

    print(f'Average accuracy across folds: {cumulative_accuracy / 5}')


if __name__ == '__main__':
    __train_classifier()
