"""Embeddings using Continuous Bag-of-Word model
Contains the `Classifier` class representing a sentiment classifier using a bag of words model
"""

from typing import Any, Dict, List, Set, Tuple, cast

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

class BagOfWords(nn.Module):
    """A PyTorch neural network class using a simple bag-of-words model"""
    __doc__ += nn.Module.__doc__  # type: ignore

    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()

        # As simple model with 2 linear layers
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 1)

    def forward(self, *input: Tensor, **kwargs: Any):
        # Sum the tensor on the first dimension to feed into linear layer
        embedding = cast(Tensor, sum(self.embedding(*input))).view(1, -1)

        output = self.linear1(embedding)
        output = self.activation_function1(output)

        # Remove unnecessary dimension at index 0
        return self.linear2(output.squeeze(0))


class Classifier():
    def __init__(self, vocabulary: Set[str], embedding_size: int):
        self.vocabulary = vocabulary
        self.vocabulary.add('<pad>')

        # Store word indices for faster lookup
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        # Determine if user has CUDA GPU and if they do, apply the model and criterion to it
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BagOfWords(len(vocabulary), embedding_size).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def load(self, model_path: str) -> None:
        """Load trained model from path

        Parameters
        ----------
        model_path (str): The path of the saved model state to load
        """

        model_dictionary: Dict[str, Any] = torch.load(model_path)
        self.model.load_state_dict(model_dictionary)

    def preprocess_data(
        self, texts: List[Any], text_labels: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Creates batches of data of the appropriate format from the given texts and labels

        Parameters
        ----------
        texts (List[Any]): Tokenized texts to use as input data for the model

        text_labels (List[int]): The corresponding labels for each text in `texts`

        Returns
        -------
        Tuple[List[Tensor], List[Tensor]]: Tuple of list of processed data and corresponding labels
        """

        data: List[Tensor] = []
        labels: List[Tensor] = []

        for index, text in enumerate(texts):
            # Create tensors from the word list and the label
            data.append(self.__create_words_vector([word.text for word in text]))
            labels.append(torch.tensor([text_labels[index]], dtype=torch.float, device=self.device))

        return data, labels

    def train(
        self, batches: List[Tensor], labels: List[Tensor], optimizer: Optimizer
    ) -> Tuple[float, float]:
        """Trains the classifier for one epoch and returns the training metrics

        Parameters
        ----------
        batches (List[Tensor]): Batches of data tensors used to train the model

        labels (List[Tensor]): The corresponding label tensors for each batch

        optimizer (Optimizer): Optimizer to use for training

        Returns
        -------
        Tuple[float, float]: Tuple of mean training loss and accuracy for the current epoch
        """

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        self.model.train()
        # Show the progress of the training
        for data, label in cast(Any, tqdm(zip(batches, labels), total=len(batches))):
            # Zero the gradients for the current backpropagation pass
            optimizer.zero_grad()

            # Do prediction using the model and get the resulting loss and accuracy
            loss, accuracy = self.__predict(data, label)

            # Do backpropagation with the loss and advance optimizer
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

        # Calculate mean loss and accuracy from the accumulated values
        return epoch_loss / len(batches), epoch_accuracy / len(batches)

    def evaluate(self, batches: List[Tensor], labels: List[Tensor]) -> Tuple[float, float]:
        """Evaluates the classifier for the current epoch and returns the validation metrics

        Parameters
        ----------
        batches (List[Tensor]): Batches of data tensors used to evaluate the model

        labels (List[Tensor]): The corresponding label tensors for each batch

        Returns
        -------
        Tuple[float, float]: Tuple of mean validation loss and accuracy for the current epoch
        """

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        self.model.eval()

        # Not training the model here, so we don't need gradients
        with torch.no_grad():
            for data, label in zip(batches, labels):
                # Do prediction using the model and get the resulting loss and accuracy
                loss, accuracy = self.__predict(data, label)

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()

        # Calculate mean loss and accuracy from the accumulated values
        return epoch_loss / len(batches), epoch_accuracy / len(batches)

    def predict(self, text: List[str]) -> int:
        """Makes a polarity prediction on the given text using the trained model

        Parameters
        ----------
        text (List[str]): The text to classify as positive or negative

        Returns
        -------
        int: The predicted polarity of the text (0: positive, 1: negative)
        """

        context_vector = self.__create_words_vector(text)
        return int(torch.round(torch.sigmoid(context_vector)).item())

    # Private methods
    def __create_words_vector(self, words: List[str]) -> Tensor:
        word_indices = [self.word_to_index[word] for word in words]
        return torch.tensor(word_indices, dtype=torch.long, device=self.device)

    def __predict(self, data: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        self.model.zero_grad()
        predictions: Tensor = self.model(data)
        loss = self.criterion(predictions, label)

        return loss, torch.round(torch.sigmoid(predictions)) == label
