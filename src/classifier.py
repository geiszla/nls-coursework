"""Embeddings using Continuous Bag-of-Word model
Contains the `Embeddings` class representing word embeddings using a trainable CBOW model.

> Note: Only Proof-of-concept, not used in final program. Not tested, probably doesn't even work
"""

from math import ceil
from statistics import mean
import time
from typing import Any, Dict, List, Set, Tuple, cast

import numpy
from sklearn.model_selection import KFold
import torch
from torch import Tensor
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocabulary_size: int, feature_size: int):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, feature_size)
        self.rnn = nn.RNN(feature_size, 256)
        self.classify = nn.Linear(256, 1)

    def forward(self, *input: Tensor, **kwargs: Any):
        embedding = self.embedding(*input)
        output, hidden = self.rnn(embedding)

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.classify(hidden.squeeze(0))


class Classifier():
    def __init__(self, vocabulary: Set[str], feature_size: int, context_size: int):
        self.context_size = context_size

        self.vocabulary = vocabulary
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        self.model = RNN(len(vocabulary), feature_size)
        self.criterion = nn.BCEWithLogitsLoss()

    def load(self, model_path: str) -> None:
        model_dictionary: Dict[str, Any] = torch.load(model_path)
        self.model.load_state_dict(model_dictionary)

    def preprocess_data(self, texts: List[str]) -> List[Tensor]:
        data: List[Tensor] = []

        for text in texts:
            words = text.split()

            for index in range((self.context_size - 1), len(words) - (self.context_size - 1)):
                start_index = index - self.context_size // 2
                start_index = start_index if start_index >= 0 else 0
                end_index = ceil(index + self.context_size / 2)
                end_index = end_index if end_index < len(words) else len(words) - 1

                context = [words[index] for index in range(start_index, end_index + 1)]
                data.append(self.__create_context_vector(context))

        return data

    def train(self, batches: List[Tensor], labels: List[Tensor]) -> Tuple[float, float]:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        self.model.train()
        for data, label in zip(batches, labels):
            optimizer.zero_grad()

            loss, accuracy = self.__predict(data, label)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

        return epoch_loss / len(batches), epoch_accuracy / len(batches)

    def evaluate(self, batches: List[Tensor], labels: List[Tensor]) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_accuracy = 0

        self.model.eval()

        with torch.no_grad():
            for data, label in zip(batches, labels):
                loss, accuracy = self.__predict(data, label)

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()

        return epoch_loss / len(batches), epoch_accuracy / len(batches)

    def test(self, data: List[Tensor], labels: List[Tensor]) -> float:
        kfold = KFold(n_splits=10)

        accuracies: List[float] = []
        for train_indices, test_indices in cast(
            List[Tuple[List[int], List[int]]], kfold.split(data)
        ):
            training_data = numpy.array(data)[train_indices]
            training_labels = numpy.array(labels)[train_indices]
            evaluation_data = numpy.array(data)[test_indices]
            evaluation_labels = numpy.array(labels)[test_indices]

            best_valid_loss = float('inf')

            for epoch in range(50):
                start_time = time.time()

                training_loss, training_accuracy = self.train(training_data, training_labels)
                validation_loss, validation_accuracy = self.evaluate(
                    evaluation_data,
                    evaluation_labels
                )

                epoch_mins = time.time() - start_time
                epoch_secs = int(epoch_mins - (int(epoch_mins / 60) * 60))

                if validation_loss < best_valid_loss:
                    best_valid_loss = validation_loss
                    accuracies.append(validation_accuracy)

                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {training_loss:.3f}'
                    + f' | Train Acc: {training_accuracy * 100:.2f}%')
                print(f'\t Val. Loss: {validation_loss:.3f}'
                    + f' |  Val. Acc: {validation_accuracy * 100:.2f}%')

        return mean(accuracies)

    def predict(self, context: List[str]) -> numpy.ndarray:
        context_vector = self.__create_context_vector(context)
        return cast(numpy.ndarray, self.model(context_vector).data.numpy())

    # Private methods
    def __create_context_vector(self, context: List[str]) -> Tensor:
        word_indexes = [self.word_to_index[word] for word in context]

        return torch.tensor(word_indexes, dtype=torch.long)

    def __predict(self, data: Tensor, label: Tensor):
        predictions: Any = self.model(data).squeeze(1)
        loss = self.criterion(predictions, label)

        rounded_predictions = torch.round(torch.sigmoid(predictions))
        correct = (rounded_predictions == label).float()
        accuracy = correct.sum() / len(correct)

        loss.backward()
        return loss, accuracy
