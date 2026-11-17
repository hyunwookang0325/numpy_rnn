"""Tokenizers and data loading for the character-level name dataset (data/names.txt)."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "names.txt"

LETTER_TO_NUM: dict[str, int] = {
    "BOS_ID": 1,
    "PAD_ID": 0,
    "EOS_ID": 2,
    "IGNORE_ID": -1,
    "a": 3,
    "b": 4,
    "c": 5,
    "d": 6,
    "e": 7,
    "f": 8,
    "g": 9,
    "h": 10,
    "i": 11,
    "j": 12,
    "k": 13,
    "l": 14,
    "m": 15,
    "n": 16,
    "o": 17,
    "p": 18,
    "q": 19,
    "r": 20,
    "s": 21,
    "t": 22,
    "u": 23,
    "v": 24,
    "w": 25,
    "x": 26,
    "y": 27,
    "z": 28,
}

NUM_TO_LETTER: dict[int, str] = {
    0: "PAD_ID",
    1: "BOS_ID",
    2: "EOS_ID",
    3: "a",
    4: "b",
    5: "c",
    6: "d",
    7: "e",
    8: "f",
    9: "g",
    10: "h",
    11: "i",
    12: "j",
    13: "k",
    14: "l",
    15: "m",
    16: "n",
    17: "o",
    18: "p",
    19: "q",
    20: "r",
    21: "s",
    22: "t",
    23: "u",
    24: "v",
    25: "w",
    26: "x",
    27: "y",
    28: "z",
    -1: "IGNORE_ID",
}


def input_tokenizer(
    text_list: list[str], dictionary: dict[str, int] = LETTER_TO_NUM
) -> np.ndarray:
    """
    Encode names as one-hot input sequences that start with BOS.

    Each sequence has length T = (longest name length) + 1. Time step 0 is BOS,
    letter j of a name is placed at time step j + 1, and the remaining steps are
    left as all-zero vectors (padding).

    Args:
        text_list (list[str]): Names (or other strings) to encode.
        dictionary (dict[str, int]): Mapping from token to class index.

    Returns:
        np.ndarray: One-hot inputs, shape [N, T, 29], dtype float32.
    """
    max_length: int = max([len(name) for name in text_list])
    vocab_length: int = max_length + 1
    dict_num: int = 29

    output_list: list[np.ndarray] = [
        np.zeros(shape=(vocab_length, dict_num)) for _ in range(len(text_list))
    ]
    output_array: np.ndarray = np.array(output_list, dtype=np.float32)

    for i, name in enumerate(text_list):
        output_array[i][0][1] = 1
        for j, letter in enumerate(name):
            output_array[i][j + 1][dictionary[letter]] = 1

    return output_array


def output_tokenizer(
    text_list: list[str], dictionary: dict[str, int] = LETTER_TO_NUM
) -> np.ndarray:
    """
    Encode names as target sequences of class indices that end with EOS.

    Each sequence has length T = (longest name length) + 1, matching
    input_tokenizer. Letter j of a name is placed at time step j (no shift), EOS
    follows the last letter, and the remaining steps are filled with -1 so they
    are skipped by nn.CrossEntropyLoss(ignore_index=-1).

    Args:
        text_list (list[str]): Names (or other strings) to encode.
        dictionary (dict[str, int]): Mapping from token to class index.

    Returns:
        np.ndarray: Target class indices, shape [N, T], dtype int64.
    """
    max_length: int = max([len(name) for name in text_list])
    vocab_length: int = max_length + 1
    output_list: list[np.ndarray] = [
        np.ones(shape=(vocab_length,), dtype=np.int64) * -1
        for _ in range(len(text_list))
    ]
    output_array: np.ndarray = np.array(output_list, dtype=np.int64)

    for i, name in enumerate(text_list):
        for j, letter in enumerate(name):
            output_array[i][j] = dictionary[letter]
        output_array[i][j + 1] = dictionary["EOS_ID"]
    return output_array


def set_input_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Read data/names.txt and encode every name into model inputs and targets.

    Returns:
        tuple[np.ndarray, np.ndarray]: (input_token, output_token), where
            input_token has shape [N, T, 29] (float32, one-hot) and
            output_token has shape [N, T] (int64, -1 = padding).
    """
    input_text: list[str] = DATA.read_text().splitlines()
    input_token: np.ndarray = input_tokenizer(input_text)
    output_token: np.ndarray = output_tokenizer(input_text)

    return input_token, output_token


def set_train_val_set(
    input_token: np.ndarray,
    output_token: np.ndarray,
    ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split encoded inputs and targets into training and validation sets.

    Args:
        input_token (np.ndarray): One-hot inputs, shape [N, T, 29].
        output_token (np.ndarray): Target class indices, shape [N, T].
        ratio (float): Fraction of sequences used for training.
        random_state (int): Seed for the shuffle before splitting.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (X_train, X_val, y_train, y_val).
    """
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        input_token, output_token, train_size=ratio, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last axis (the row max is subtracted first).

    Args:
        logits (np.ndarray): Unnormalized scores, shape [..., V].

    Returns:
        np.ndarray: Probabilities that sum to 1 along the last axis, same shape as logits.
    """
    logits_max: np.ndarray = logits.max(axis=-1, keepdims=True)
    logits = logits - np.ones(shape=logits.shape, dtype=np.float32) * logits_max
    nominator = np.exp(logits)
    denominator = nominator.sum(axis=-1, keepdims=True)
    return nominator / denominator


def convert_np_to_torch(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Wrap NumPy train/validation arrays in TensorDatasets and return DataLoaders.

    Inputs are cast to float32. Training targets are cast to int64, as required by
    nn.CrossEntropyLoss.

    Args:
        X_train (np.ndarray): Training inputs, one-hot, shape [N_train, T, V].
        X_val (np.ndarray): Validation inputs, one-hot, shape [N_val, T, V].
        y_train (np.ndarray): Training targets (class indices, -1 = padding),
            shape [N_train, T].
        y_val (np.ndarray): Validation targets (class indices, -1 = padding),
            shape [N_val, T].
        batch_size (int): Number of sequences per batch.
        shuffle (bool): Whether to shuffle the training set every epoch.
            The validation set is never shuffled.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader).
    """
    X_train: torch.Tensor = torch.as_tensor(X_train, dtype=torch.float32)
    X_val: torch.Tensor = torch.as_tensor(X_val, dtype=torch.float32)
    y_train: torch.Tensor = torch.as_tensor(y_train, dtype=torch.int64)
    y_val: torch.Tensor = torch.as_tensor(y_val)

    train_data: TensorDataset = TensorDataset(X_train, y_train)
    val_data: TensorDataset = TensorDataset(X_val, y_val)

    train_loader: DataLoader = DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle
    )
    val_loader: DataLoader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader


def CrossEntropyLoss(
    yhat: np.ndarray, y_true: np.ndarray, ignore_index: int = -100
) -> tuple[float, np.ndarray]:
    """
    Mean cross-entropy over non-ignored targets, and its gradient w.r.t. the logits.

    Matches ``torch.nn.CrossEntropyLoss(ignore_index=...)`` with mean reduction.
    The gradient uses softmax + cross-entropy combined: (p - one_hot) / n_valid,
    with ignored rows set to 0.

    Args:
        yhat (np.ndarray): Softmax probabilities (not logits), shape [N, V].
        y_true (np.ndarray): Target class indices, shape [N].
        ignore_index (int): Target value to skip (padding).

    Returns:
        tuple[float, np.ndarray]: (loss, dlogits), where dlogits = dLoss/dlogits
            has shape [N, V].
    """
    mask: np.ndarray = y_true != ignore_index
    tmp: np.ndarray = y_true.copy()
    y_true_: np.ndarray = np.where(mask, tmp, 0)
    y_true_one_hot = np.eye(yhat.shape[1])[y_true_]

    loss_array = -np.sum(np.log(yhat) * y_true_one_hot, axis=1)
    loss = np.mean(loss_array[mask])

    dlogits = yhat - y_true_one_hot
    dlogits = np.where(mask[:, None] == 1, dlogits, 0)
    column_true = mask.sum(axis=0)
    dlogits /= column_true
    return loss, dlogits


def main() -> None:
    """Encode names.txt, split it, and log a sample token and the array shapes."""
    from loguru import logger

    input_text: list[str] = DATA.read_text().splitlines()
    input_token: np.ndarray = input_tokenizer(input_text)
    output_token: np.ndarray = output_tokenizer(input_text)
    logger.info(f"input token [0]: {input_token[0][0]}")
    logger.info(f"output token [0]: {output_token[0]}")
    X_train, X_val, y_train, y_val = set_train_val_set(input_token, output_token)
    logger.info(f"shape of X train: {X_train.shape}")
    logger.info(f"shape of X train: {y_train.shape}")


if __name__ == "__main__":
    main()
