"""Embeddings using Continuous Bag-of-Word model
Contains the `Embeddings` class representing word embeddings using a trainable CBOW model.

> Proof-of-concept, not used in final program
"""

from math import ceil
from random import random
from typing import cast, List, Tuple

import numpy
from nptyping import Array
from sklearn.cluster import KMeans
import torch
from torch import Tensor
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocabulary_size: int, feature_size: int):
        super().__init__()

        self.embeddings = nn.Embedding(vocabulary_size, feature_size)
        self.linear1 = nn.Linear(feature_size, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocabulary_size)
        self.activation_function2 = nn.LogSoftmax(-1)

    def forward(self, inputs: Tensor):  # type: ignore # pylint: disable=arguments-differ
        embedding = cast(Tensor, sum(self.embeddings(inputs))).view(1, -1)
        output = self.linear1(embedding)
        output = self.activation_function1(output)
        output = self.linear2(output)
        output = self.activation_function2(output)

        return output

    def get_embedding(self, word_index: int) -> Array[float]:
        word_tensor = torch.LongTensor([word_index])  # type: ignore
        return self.embeddings(word_tensor).view(1, -1)


class Embeddings():
    def __init__(self, vocabulary: List[str], feature_size: int, context_size: int):
        self.vocabulary = vocabulary
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.model = CBOW(len(vocabulary), feature_size)

        self.context_size = context_size

    def train(self, texts: List[str]) -> None:
        data: List[Tuple[Tensor, str]] = []

        for text in texts:
            words = text.split()

            for index in range(2, len(words) - 2):
                start_index = index - self.context_size // 2
                start_index = start_index if start_index >= 0 else 0
                end_index = ceil(index + self.context_size / 2)
                end_index = end_index if end_index < len(words) else len(words) - 1

                context = [words[index] for index in range(start_index, end_index + 1)]
                target = text[index]

                data.append((self.__create_context_vector(context), target))

        loss_function = nn.NLLLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        for _ in range(50):
            total_loss = 0.0

            for context_vector, target in data:
                self.model.zero_grad()

                log_probabilities = self.model(context_vector)
                loss = loss_function(
                    log_probabilities,
                    torch.tensor(  # pylint: disable=not-callable
                        [self.word_to_index[target]],
                        dtype=torch.long
                    )
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def predict(self, context: List[str]) -> str:
        context_vector = self.__create_context_vector(context)
        [probabilities] = self.model(context_vector).data.numpy()

        most_probable_index = probabilities.index(max(probabilities))
        return self.vocabulary[most_probable_index]

    def cluster(self, words: List[str], cluster_count: int) -> Array[numpy.int32]:
        word_embeddings = [self.model.get_embedding(self.word_to_index[word]) for word in words]
        return KMeans(cluster_count).fit_predict(word_embeddings)

    def test(self, cluster_count: int) -> None:
        target_words = [word[::-1] for word in self.vocabulary if random() > 0.5]

        cluster_count = len(self.vocabulary)
        original_clusters = self.cluster(target_words, cluster_count)
        reversed_clusters = self.cluster(self.vocabulary, cluster_count)

        correct_count = sum(1 for index, cluster in enumerate(original_clusters)
            if cluster == reversed_clusters[index])

        print(f'Correct pairs: {correct_count/cluster_count}')

    def __create_context_vector(self, context: List[str]) -> Tensor:
        word_indexes = [self.word_to_index[word] for word in context]

        return torch.tensor(  # pylint: disable=not-callable
            word_indexes,
            dtype=torch.long
        )
