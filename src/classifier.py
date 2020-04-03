"""Embeddings using Continuous Bag-of-Word model
Contains the `Embeddings` class representing word embeddings using a trainable CBOW model.

> Note: Only Proof-of-concept, not used in final program. Not tested, probably doesn't even work
"""

from typing import Any, Dict, List, Set, Tuple, cast

import numpy
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from tqdm import tqdm

class BOW(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 1)

    def forward(self, *input: Tensor, **kwargs: Any):
        embedding = cast(Tensor, sum(self.embedding(*input))).view(1, -1)

        output = self.linear1(embedding)
        output = self.activation_function1(output)

        return self.linear2(output.squeeze(0))


class Classifier():
    def __init__(self, vocabulary: Set[str], embedding_size: int):
        self.vocabulary = vocabulary
        self.vocabulary.add('<pad>')

        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BOW(len(vocabulary), embedding_size).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def load(self, model_path: str) -> None:
        model_dictionary: Dict[str, Any] = torch.load(model_path)
        self.model.load_state_dict(model_dictionary)

    def preprocess_data(
        self, texts: List[Any], text_labels: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        data: List[Tensor] = []
        labels: List[Tensor] = []

        for index, text in enumerate(texts):
            data.append(self.__create_words_vector([word.text for word in text]))
            labels.append(torch.tensor([text_labels[index]], dtype=torch.float, device=self.device))

        return data, labels

    def train(
        self, batches: List[Tensor], labels: List[Tensor], optimizer: Optimizer
    ) -> Tuple[float, float]:
        epoch_loss = 0.0
        epoch_accuracy = 0

        self.model.train()
        for data, label in cast(Any, tqdm(zip(batches, labels), total=len(batches))):
            optimizer.zero_grad()

            loss, accuracy = self.__predict(data, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

        return epoch_loss / len(batches), epoch_accuracy / len(batches)

    def evaluate(self, batches: List[Tensor], labels: List[Tensor]) -> Tuple[float, float]:
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        self.model.eval()

        with torch.no_grad():
            for data, label in zip(batches, labels):
                loss, accuracy = self.__predict(data, label)

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()

        return epoch_loss / len(batches), epoch_accuracy / len(batches)

    def predict(self, words: List[str]) -> numpy.ndarray:
        context_vector = self.__create_words_vector(words)
        return cast(numpy.ndarray, self.model(context_vector).data.numpy())

    # Private methods
    def __create_words_vector(self, words: List[str]) -> Tensor:
        word_indices = [self.word_to_index[word] for word in words]
        return torch.tensor(word_indices, dtype=torch.long, device=self.device)

    def __predict(self, data: Tensor, label: Tensor):
        self.model.zero_grad()
        predictions: Tensor = self.model(data)
        loss = self.criterion(predictions, label)

        return loss, torch.round(torch.sigmoid(predictions)) == label
