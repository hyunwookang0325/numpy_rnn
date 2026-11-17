"""Character-level language models (RNN / GRU / LSTM) and a training loop in PyTorch."""

from typing import Any

import torch
import torch.nn as nn

# import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

# import numpy as np
from .utils import convert_np_to_torch, set_input_data, set_train_val_set


class RNN(nn.Module):
    """
    Vanilla (Elman) RNN followed by a linear classifier applied at every time step.

    Attributes:
        hidden (torch.Tensor | None): Last hidden state h_n from the latest forward pass,
            shape [1, B, hidden_size].
    """

    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ) -> None:
        """
        Args:
            input_size (int): Size of each input vector (vocabulary size, one-hot).
                Also used as the number of output classes.
            hidden_size (int): Number of features in the hidden state.
            batch_first (bool): If True, inputs are shaped [B, T, input_size].
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.rnn: nn.RNN = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.classifier: nn.Linear = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )
        self.hidden: torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the recurrent layer over the whole sequence and project every time step to logits.

        Args:
            X (torch.Tensor): One-hot input, shape [B, T, input_size].

        Returns:
            torch.Tensor: Logits flattened over batch and time, shape [B * T, input_size].
        """
        output, self.hidden = self.rnn(X)
        out = self.classifier(output)
        logits = out.view(-1, self.input_size)
        return logits


class GRU(nn.Module):
    """
    GRU followed by a linear classifier applied at every time step.

    Attributes:
        hidden (torch.Tensor | None): Last hidden state h_n from the latest forward pass,
            shape [1, B, hidden_size].
    """

    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ) -> None:
        """
        Args:
            input_size (int): Size of each input vector (vocabulary size, one-hot).
                Also used as the number of output classes.
            hidden_size (int): Number of features in the hidden state.
            batch_first (bool): If True, inputs are shaped [B, T, input_size].
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.gru: nn.GRU = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.classifier: nn.Linear = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )
        self.hidden: torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the recurrent layer over the whole sequence and project every time step to logits.

        Args:
            X (torch.Tensor): One-hot input, shape [B, T, input_size].

        Returns:
            torch.Tensor: Logits flattened over batch and time, shape [B * T, input_size].
        """
        output, self.hidden = self.gru(X)
        out = self.classifier(output)
        logits = out.view(-1, self.input_size)
        return logits


class LSTM(nn.Module):
    """
    LSTM followed by a linear classifier applied at every time step.

    Attributes:
        hidden (torch.Tensor | None): Last hidden state h_n from the latest forward pass,
            shape [1, B, hidden_size].
        cell (torch.Tensor | None): Last cell state c_n from the latest forward pass,
            shape [1, B, hidden_size].
    """

    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ) -> None:
        """
        Args:
            input_size (int): Size of each input vector (vocabulary size, one-hot).
                Also used as the number of output classes.
            hidden_size (int): Number of features in the hidden state.
            batch_first (bool): If True, inputs are shaped [B, T, input_size].
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.classifier: nn.Linear = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )
        self.hidden: torch.Tensor | None = None
        self.cell: torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the recurrent layer over the whole sequence and project every time step to logits.

        Args:
            X (torch.Tensor): One-hot input, shape [B, T, input_size].

        Returns:
            torch.Tensor: Logits flattened over batch and time, shape [B * T, input_size].
        """
        output, (self.hidden, self.cell) = self.lstm(X)
        out = self.classifier(output)
        logits = out.view(-1, self.input_size)
        return logits


def train(
    model_type: str = "gru",
    n_epochs: int = 200,
    learning_rate: float = 0.01,
    device: str = "cpu",
    input_size: int = 29,
    hidden_size: int = 16,
    batch_first: bool = True,
    optimizer_type: str = "Adam",
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    output_name: str = "trained_best",
) -> None:
    """
    Train a recurrent language model and save the checkpoint with the lowest validation loss.

    Each epoch runs one pass over the training set and one over the validation set.
    Epoch losses are averaged over valid (non-padding) tokens. Whenever the validation
    loss reaches a new minimum, a checkpoint is written to "{output_name}.pth"
    (relative to the current working directory).

    Args:
        model_type (str): Model to train: "rnn", "gru", or "lstm".
        n_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device the batches are moved to (e.g. "cpu", "cuda").
        input_size (int): Vocabulary size (input features and output classes).
        hidden_size (int): Hidden state size of the recurrent layer.
        batch_first (bool): If True, inputs are shaped [B, T, input_size].
        optimizer_type (str): Optimizer to use: "Adam" or "SGD".
        train_loader (DataLoader | None): Training batches of (X, y).
        val_loader (DataLoader | None): Validation batches of (X, y).
        output_name (str): Checkpoint path without the ".pth" extension.

    The saved checkpoint is a dict with keys "epoch", "model_state_dict",
    "optimizer_state_dict", "train_loss", and "val_loss".
    """
    if model_type == "rnn":
        model: RNN | GRU | LSTM = RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
    elif model_type == "gru":
        model = GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
    elif model_type == "lstm":
        model = LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
    else:
        model = GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )

    if optimizer_type == "Adam":
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function always be CrossEntropyLoss
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)

    model_dict: dict[str, Any] = {
        "epoch": None,
        "model_state_dict": None,
        "train_loss": None,
        "val_loss": None,
        "optimizer_state_dict": None,
    }

    train_losses: list[float] = []
    val_losses: list[float] = []
    logger.info("Start to train")

    for epoch in range(n_epochs):
        model.train()
        total_loss: float = 0.0
        count: int = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            yhat: torch.Tensor = model(X_batch)
            # logger.info(yhat.shape)
            y_true: torch.Tensor = y_batch.view(-1)
            loss: torch.Tensor = loss_fn(yhat, y_true)
            n_valid = y_batch != -1
            n_valid = n_valid.sum().item()
            total_loss += loss.item() * n_valid
            count += n_valid

            loss.backward()
            optimizer.step()

        train_loss: float = total_loss / count

        total_loss = 0.0
        count = 0

        model.eval()
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                yhat = model(X_batch)
                y_true = y_batch.view(-1)
                loss = loss_fn(yhat, y_true)
                n_valid = y_batch != -1
                n_valid = n_valid.sum().item()

                total_loss += loss.item() * n_valid
                count += n_valid

        val_loss: float = total_loss / count

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss <= min(val_losses):
            model_dict["epoch"] = epoch
            model_dict["model_state_dict"] = model.state_dict()
            model_dict["optimizer_state_dict"] = optimizer.state_dict()
            model_dict["train_loss"] = train_loss
            model_dict["val_loss"] = val_loss

            torch.save(model_dict, f"{output_name}.pth")

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"{epoch + 1} has Train loss: {train_loss} val loss: {val_loss}"
            )

    logger.info("Train successfully finished")


def main() -> None:
    """Load names.txt, split it into train/validation sets, and run a short training job."""
    from loguru import logger

    input_token, output_token = set_input_data()
    X_train, X_val, y_train, y_val = set_train_val_set(
        input_token, output_token, ratio=0.8, random_state=42
    )
    logger.info(X_train.shape)
    train_loader, val_loader = convert_np_to_torch(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        batch_size=16,
        shuffle=True,
    )
    train(n_epochs=1, train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
