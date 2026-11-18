"""Character-level RNN / GRU / LSTM language models written in NumPy (forward, BPTT backward, SGD).

The weights are initialized from ``torch.nn.RNN`` / ``torch.nn.GRU`` / ``torch.nn.LSTM`` /
``torch.nn.Linear``
so that every NumPy result can be compared 1:1 with PyTorch autograd. PyTorch is used only
for weight initialization and ``DataLoader`` batching; all math is done in NumPy.

Shape symbols used throughout:
    B: batch size, T: sequence length, V: vocabulary size (29), H: hidden size.
"""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from .utils import (
    CrossEntropyLoss,
    convert_np_to_torch,
    set_input_data,
    set_train_val_set,
    softmax,
)


class RNN(nn.Module):
    """
    Vanilla (Elman) RNN layer plus a linear classifier, implemented in NumPy.

    ``h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)``, then
    ``logits_t = h_t W_fc^T + b_fc``.

    ``forward`` / ``backward`` / ``step`` cover the recurrent layer only. The classifier
    (fc), softmax, cross-entropy and the fc update are done inside ``train``.

    Attributes:
        weight_ih (np.ndarray): Input-to-hidden weight, shape [H, V].
        weight_hh (np.ndarray): Hidden-to-hidden weight, shape [H, H].
        bias_ih (np.ndarray): Input-to-hidden bias, shape [H].
        bias_hh (np.ndarray): Hidden-to-hidden bias, shape [H].
        weight_fc (np.ndarray): Classifier weight, shape [V, H].
        bias_fc (np.ndarray): Classifier bias, shape [V].
        cache (dict[str, np.ndarray]): Per-time-step values saved by ``forward`` for
            ``backward``. Keys: "x_t" [B, T, V], "h_t", "tx", "th", "h_prev" [B, T, H].
        diff (dict[str, np.ndarray | None]): Gradients from the latest ``backward``.
        PARAMS (list[str]): Names of the trainable arrays saved to the npz checkpoint.
    """

    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ) -> None:
        """
        Args:
            input_size (int): Vocabulary size V (one-hot input size and number of
                output classes).
            hidden_size (int): Hidden state size H.
            batch_first (bool): If True, inputs are shaped [B, T, V].
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first
        self.rnn: nn.RNN = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=self.batch_first,
        )
        rnn_state_dict: dict[str, torch.Tensor] = self.rnn.state_dict()
        self.weight_ih: np.ndarray = (
            rnn_state_dict["weight_ih_l0"].detach().cpu().numpy()
        )
        self.weight_hh: np.ndarray = (
            rnn_state_dict["weight_hh_l0"].detach().cpu().numpy()
        )
        self.bias_ih: np.ndarray = rnn_state_dict["bias_ih_l0"].detach().cpu().numpy()
        self.bias_hh: np.ndarray = rnn_state_dict["bias_hh_l0"].detach().cpu().numpy()

        self.classifier: nn.Linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.input_size
        )
        fc_state_dict: dict[str, torch.Tensor] = self.classifier.state_dict()
        self.weight_fc: np.ndarray = fc_state_dict["weight"].detach().cpu().numpy()
        self.bias_fc: np.ndarray = fc_state_dict["bias"].detach().cpu().numpy()

        self.hidden: np.ndarray | None = None
        self.cache: dict[str, np.ndarray] = {}
        self.diff: dict[str, np.ndarray | None] = {}
        self.PARAMS: list[str] = [
            "weight_fc",
            "bias_fc",
            "weight_ih",
            "weight_hh",
            "bias_ih",
            "bias_hh",
        ]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the RNN over the whole sequence, starting from a zero hidden state.

        Args:
            X (np.ndarray): One-hot input, shape [B, T, V].

        Returns:
            tuple[np.ndarray, np.ndarray]: (output, h_last), where output holds the
                hidden state of every time step, shape [B, T, H], and h_last is the
                hidden state of the last time step, shape [B, H].
        """
        B, T, V = X.shape
        H: int = self.hidden_size
        self.initial_hidden: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)
        h_prev: np.ndarray = self.initial_hidden.copy()

        self.cache = {
            "x_t": np.zeros(shape=(B, T, V), dtype=np.float32),
            "h_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "tx": np.zeros(shape=(B, T, H), dtype=np.float32),
            "th": np.zeros(shape=(B, T, H), dtype=np.float32),
            "h_prev": np.zeros(shape=(B, T, H), dtype=np.float32),
        }
        # logger.info(f"B: {B}, T: {T}, H: {H}")
        # logger.info(f"weight ih: {self.weight_ih.shape}")
        # logger.info(f"bias ih: {self.bias_ih.shape}")

        for t in range(T):
            x_t = X[:, t, :]
            tx = x_t @ self.weight_ih.T + self.bias_ih
            th = h_prev @ self.weight_hh.T + self.bias_hh
            h_t = np.tanh(tx + th)
            # logger.info(f"x_t shape: {x_t.shape}")
            # logger.info(f"tx shape: {tx.shape}")
            # logger.info(f"th shape: {th.shape}")
            # logger.info(f"h_t shape: {h_t.shape}")
            self.cache["x_t"][:, t, :] = x_t
            self.cache["tx"][:, t, :] = tx
            self.cache["th"][:, t, :] = th
            self.cache["h_t"][:, t, :] = h_t
            self.cache["h_prev"][:, t, :] = h_prev
            h_prev = h_t

        # return output and last hidden state
        return self.cache["h_t"], h_t

    def backward(self, doutput: np.ndarray) -> dict[str, np.ndarray | None]:
        """
        Backpropagation through time for the RNN layer, using the latest ``forward`` cache.

        Gradients of the shared weights are summed over all time steps.

        Args:
            doutput (np.ndarray): dLoss/d(output), the gradient flowing into every
                hidden state from the classifier, shape [B, T, H].

        Returns:
            dict[str, np.ndarray | None]: Gradients (also stored in ``self.diff``):
                "dx" [B, T, V], "dweight_ih" [H, V], "dweight_hh" [H, H],
                "dbias_ih" [H], "dbias_hh" [H].
        """
        B, T, H = doutput.shape

        dx: np.ndarray = np.zeros(shape=self.cache["x_t"].shape, dtype=np.float32)
        dweight_ih: np.ndarray = np.zeros(shape=self.weight_ih.shape, dtype=np.float32)
        dweight_hh: np.ndarray = np.zeros(shape=self.weight_hh.shape, dtype=np.float32)
        dbias_ih: np.ndarray = np.zeros(shape=self.bias_ih.shape, dtype=np.float32)
        dbias_hh: np.ndarray = np.zeros(shape=self.bias_hh.shape, dtype=np.float32)

        self.diff: dict[str, np.ndarray | None] = {
            "dx": None,
            "dweight_ih": None,
            "dweight_hh": None,
            "dbias_ih": None,
            "dbias_hh": None,
        }

        dh_next: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)
        for t in reversed(range(T)):
            x_t = self.cache["x_t"][:, t, :]
            # tx = self.cache["tx"][:, t, :]
            # th = self.cache["th"][:, t, :]
            h_prev = self.cache["h_prev"][:, t, :]
            h_t = self.cache["h_t"][:, t, :]

            # h_t = tanh(x_t @ weight_ih.T + bias_ih + h_prev @ weight_hh + bias_hh)
            # dloss / dh
            dh = doutput[:, t, :] + dh_next
            # a_t = tx+th
            # dloss / dh * dh / da_t = dh * (1-h_t **2)
            da_t = dh * (1 - h_t**2)
            # a_t = x_t @ weight_ih.T + bias_ih + h_prev @ weight_hh+bias_hh
            # da_t / dweight_ih = da_t.T @ x_t
            # dloss / dweight_ih = dloss / da_t * da_t / dweight_ih = da_t.T @ x_t
            dweight_ih += da_t.T @ x_t
            dbias_ih += np.sum(da_t, axis=0)

            dweight_hh += da_t.T @ h_prev
            dbias_hh += np.sum(da_t, axis=0)

            dx[:, t, :] += da_t @ self.weight_ih
            dh_prev = da_t @ self.weight_hh
            dh_next = dh_prev

        self.diff["dx"] = dx
        self.diff["dweight_ih"] = dweight_ih
        self.diff["dweight_hh"] = dweight_hh
        self.diff["dbias_ih"] = dbias_ih
        self.diff["dbias_hh"] = dbias_hh

        return self.diff

    def step(
        self, learning_rate: float, diff: dict[str, np.ndarray | None] | None = None
    ) -> None:
        """
        Apply one in-place SGD update to the RNN layer weights (not the classifier).

        Args:
            learning_rate (float): SGD learning rate.
            diff (dict[str, np.ndarray | None] | None): Gradients from ``backward``.
                Defaults to ``self.diff``.
        """
        # set default value
        if diff is None:
            diff = self.diff

        self.weight_ih -= learning_rate * diff["dweight_ih"]
        self.weight_hh -= learning_rate * diff["dweight_hh"]
        self.bias_ih -= learning_rate * diff["dbias_ih"]
        self.bias_hh -= learning_rate * diff["dbias_hh"]

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        n_epochs: int = 1000,
        lr: float = 0.01,
        output_name: str = "numpy_rnn",
    ) -> None:
        """
        Train with plain SGD and save the weights with the lowest validation loss.

        Per batch: forward -> fc -> softmax -> cross-entropy (padding = -1 ignored)
        -> fc backward and update -> RNN backward -> RNN step. Epoch losses are
        averaged over valid (non-padding) tokens.

        Note: this shadows ``nn.Module.train(mode)``; the torch train/eval mode is
        not used here.

        Args:
            train_loader (DataLoader): Training batches of (X [B, T, V], y [B, T]).
            val_loader (DataLoader | None): Validation batches. If None, only the
                train loss is logged and no checkpoint is written.
            n_epochs (int): Number of epochs.
            lr (float): SGD learning rate.
            output_name (str): Checkpoint path without the ".npz" extension.

        The checkpoint "{output_name}.npz" holds the arrays named in ``PARAMS`` plus
        "epoch", "val_loss", "lr" and "hidden_size". It is overwritten whenever the
        validation loss reaches a new minimum.
        """
        data: dict[str, np.ndarray | None] = {
            "weight_fc": None,
            "bias_fc": None,
            "weight_ih": None,
            "weight_hh": None,
            "bias_ih": None,
            "bias_hh": None,
        }
        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch in range(n_epochs):
            total_loss: float = 0.0
            count: int = 0
            for X_train, y_train in train_loader:
                X_train, y_train = (
                    X_train.detach().cpu().numpy(),
                    y_train.detach().cpu().numpy(),
                )
                X_train = np.array(X_train, dtype=np.float32)
                y_train = np.array(y_train, dtype=np.int64)
                Batch, Time, _ = X_train.shape
                Hidden = self.hidden_size
                output, hidden = self.forward(X_train)  # [B, T, H]
                out = output @ self.weight_fc.T + self.bias_fc  # [B, T, V]
                logits = out.reshape(-1, self.input_size)  # [B*T, V]
                yhat = softmax(logits)
                y_true = y_train.reshape(-1)  # [B*T]
                loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                output = output.reshape(-1, Hidden)  # [B*T, H]
                dweight_fc = dlogits.T @ output  # [V, H]
                dbias_fc = np.sum(dlogits, axis=0)  # [V]

                # must use weight_fc before it is updated
                doutput = dlogits @ self.weight_fc  # [B*T, H]
                self.weight_fc -= lr * dweight_fc
                self.bias_fc -= lr * dbias_fc

                doutput = doutput.reshape(Batch, Time, Hidden)  # [B, T, H]
                self.backward(doutput)
                self.step(learning_rate=lr)

                n_valid = y_true != -1
                total_loss += loss * n_valid.sum()
                count += n_valid.sum()
            data["weight_fc"] = self.weight_fc.copy()
            data["bias_fc"] = self.bias_fc.copy()
            data["weight_ih"] = self.weight_ih.copy()
            data["weight_hh"] = self.weight_hh.copy()
            data["bias_ih"] = self.bias_ih.copy()
            data["bias_hh"] = self.bias_hh.copy()

            train_loss = total_loss / count
            train_losses.append(train_loss)
            if val_loader is not None:
                total_loss = 0.0
                count = 0
                for X_val, y_val in val_loader:
                    X_val, y_val = (
                        X_val.detach().cpu().numpy(),
                        y_val.detach().cpu().numpy(),
                    )
                    X_val = np.array(X_val, dtype=np.float32)
                    y_val = np.array(y_val, dtype=np.int64)
                    output, hidden = self.forward(X_val)
                    out = output @ self.weight_fc.T + self.bias_fc
                    logits = out.reshape(-1, self.input_size)
                    yhat = softmax(logits)
                    y_true = y_val.reshape(-1)
                    loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                    n_valid = y_true != -1
                    total_loss += loss * n_valid.sum()
                    count += n_valid.sum()
                val_loss = total_loss / count
                val_losses.append(val_loss)

                if val_loss <= min(val_losses):
                    np.savez(
                        f"{output_name}.npz",
                        **{p: data[p] for p in self.PARAMS},
                        epoch=epoch,
                        val_loss=val_loss,
                        lr=lr,
                        hidden_size=self.hidden_size,
                    )

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"{epoch + 1} has train loss: {train_loss: .4f} validation loss: {val_loss: .4f}"
                    )

            else:
                logger.info(f"{epoch + 1} has train loss: {train_loss: .4f}")

        if val_loader is None:
            logger.info(
                "If you do not have any validation loader, we won't save the npz file any. REMIND this"
            )


class GRU(nn.Module):
    """
    GRU layer plus a linear classifier, implemented in NumPy.

    Gate order and weight shapes follow ``torch.nn.GRU`` (chunk order r, z, n)::

        r_t = sigmoid(x_t Wxr^T + bxr + h_{t-1} Whr^T + bhr)
        z_t = sigmoid(x_t Wxz^T + bxz + h_{t-1} Whz^T + bhz)
        n_t = tanh(x_t Wxn^T + bxn + r_t * (h_{t-1} Whn^T + bhn))
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}

    ``forward`` / ``backward`` / ``step`` cover the recurrent layer only. The classifier
    (fc), softmax, cross-entropy and the fc update are done inside ``train``.

    Attributes:
        Wxr, Wxz, Wxn (np.ndarray): Input-to-hidden weights, shape [H, V].
        Whr, Whz, Whn (np.ndarray): Hidden-to-hidden weights, shape [H, H].
        bxr, bxz, bxn, bhr, bhz, bhn (np.ndarray): Biases, shape [H].
        weight_fc (np.ndarray): Classifier weight, shape [V, H].
        bias_fc (np.ndarray): Classifier bias, shape [V].
        cache (dict[str, np.ndarray]): Per-time-step values saved by ``forward`` for
            ``backward``. Keys: "x_t" [B, T, V], "h_prev", "h_t", "r_t", "z_t",
            "n_t", "thn_t" [B, T, H].
        diff (dict[str, np.ndarray]): Gradients from the latest ``backward``.
        PARAMS (list[str]): Names of the trainable arrays saved to the npz checkpoint.
    """

    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ) -> None:
        """
        Args:
            input_size (int): Vocabulary size V (one-hot input size and number of
                output classes).
            hidden_size (int): Hidden state size H.
            batch_first (bool): If True, inputs are shaped [B, T, V].
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first
        self.gru: nn.GRU = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=self.batch_first,
        )
        gru_state_dict: dict[str, torch.Tensor] = self.gru.state_dict()
        # logger.info(gru_state_dict.keys())
        Wxr, Wxz, Wxn = gru_state_dict["weight_ih_l0"].chunk(3, 0)
        Whr, Whz, Whn = gru_state_dict["weight_hh_l0"].chunk(3, 0)
        bxr, bxz, bxn = gru_state_dict["bias_ih_l0"].chunk(3, 0)
        bhr, bhz, bhn = gru_state_dict["bias_hh_l0"].chunk(3, 0)
        self.Wxr, self.Wxz, self.Wxn = (
            Wxr.detach().cpu().numpy(),
            Wxz.detach().cpu().numpy(),
            Wxn.detach().cpu().numpy(),
        )
        self.Whr, self.Whz, self.Whn = (
            Whr.detach().cpu().numpy(),
            Whz.detach().cpu().numpy(),
            Whn.detach().cpu().numpy(),
        )
        self.bxr, self.bxz, self.bxn = (
            bxr.detach().cpu().numpy(),
            bxz.detach().cpu().numpy(),
            bxn.detach().cpu().numpy(),
        )
        self.bhr, self.bhz, self.bhn = (
            bhr.detach().cpu().numpy(),
            bhz.detach().cpu().numpy(),
            bhn.detach().cpu().numpy(),
        )

        self.classifier: nn.Linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.input_size
        )
        state_dict: dict[str, torch.Tensor] = self.classifier.state_dict()
        self.weight_fc: np.ndarray = state_dict["weight"].detach().cpu().numpy()
        self.bias_fc: np.ndarray = state_dict["bias"].detach().cpu().numpy()

        self.cache: dict[str, np.ndarray] = {}
        self.diff: dict[str, np.ndarray] = {}
        self.hidden: np.ndarray | None = None
        self.PARAMS: list[str] = [
            "weight_fc",
            "bias_fc",
            "Wxr",
            "Whr",
            "bxr",
            "bhr",
            "Wxz",
            "Whz",
            "bxz",
            "bhz",
            "Wxn",
            "Whn",
            "bxn",
            "bhn",
        ]

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Element-wise logistic sigmoid.

        Args:
            Z (np.ndarray): Pre-activation, any shape.

        Returns:
            np.ndarray: 1 / (1 + exp(-Z)), same shape as Z.
        """
        return 1 / (1 + np.exp(-Z))

    def reset_gate(self, X: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """
        Reset gate r_t.

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            hidden (np.ndarray): Previous hidden state, shape [B, H].

        Returns:
            np.ndarray: r_t, shape [B, H].
        """
        txr = X @ self.Wxr.T + self.bxr
        thr = hidden @ self.Whr.T + self.bhr
        r = self.sigmoid(txr + thr)
        return r

    def update_gate(self, X: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """
        Update gate z_t.

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            hidden (np.ndarray): Previous hidden state, shape [B, H].

        Returns:
            np.ndarray: z_t, shape [B, H].
        """
        txz = X @ self.Wxz.T + self.bxz
        thz = hidden @ self.Whz.T + self.bhz
        z = self.sigmoid(txz + thz)
        return z

    def candidate_n(
        self, X: np.ndarray, hidden: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Candidate hidden state n_t (bhn sits inside the reset gate, as in torch).

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            hidden (np.ndarray): Previous hidden state, shape [B, H].
            r (np.ndarray): Reset gate r_t, shape [B, H].

        Returns:
            tuple[np.ndarray, np.ndarray]: (n_t, thn_t), where
                thn_t = hidden @ Whn^T + bhn is kept for the backward pass.
                Both have shape [B, H].
        """
        txn = X @ self.Wxn.T + self.bxn
        thn = hidden @ self.Whn.T + self.bhn
        n = np.tanh(txn + r * thn)
        return n, thn

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the GRU over the whole sequence, starting from a zero hidden state.

        Args:
            X (np.ndarray): One-hot input, shape [B, T, V].

        Returns:
            tuple[np.ndarray, np.ndarray]: (output, h_last), where output holds the
                hidden state of every time step, shape [B, T, H], and h_last is the
                hidden state of the last time step, shape [B, H].
        """
        B, T, V = X.shape
        H = self.hidden_size

        self.cache = {
            "x_t": np.zeros(shape=(B, T, V), dtype=np.float32),
            "h_prev": np.zeros(shape=(B, T, H), dtype=np.float32),
            "h_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "r_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "z_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "n_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "thn_t": np.zeros(shape=(B, T, H), dtype=np.float32),
        }
        h_prev: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]
            r_t = self.reset_gate(X=x_t, hidden=h_prev)
            z_t = self.update_gate(X=x_t, hidden=h_prev)
            n_t, thn_t = self.candidate_n(X=x_t, hidden=h_prev, r=r_t)
            h_t = (1 - z_t) * n_t + z_t * h_prev
            self.cache["x_t"][:, t, :] = x_t
            self.cache["h_prev"][:, t, :] = h_prev
            self.cache["h_t"][:, t, :] = h_t
            self.cache["r_t"][:, t, :] = r_t
            self.cache["z_t"][:, t, :] = z_t
            self.cache["n_t"][:, t, :] = n_t
            self.cache["thn_t"][:, t, :] = thn_t
            h_prev = h_t

        # return output and lst hidden state
        return self.cache["h_t"], h_t

    def backward(self, doutput: np.ndarray) -> dict[str, np.ndarray]:
        """
        Backpropagation through time for the GRU layer, using the latest ``forward`` cache.

        Gradients of the shared weights are summed over all time steps. dh_prev
        collects four paths: through r_t, z_t, thn_t (via Whn), and the direct
        path dh * z_t.

        Args:
            doutput (np.ndarray): dLoss/d(output), the gradient flowing into every
                hidden state from the classifier, shape [B, T, H].

        Returns:
            dict[str, np.ndarray]: Gradients (also stored in ``self.diff``):
                "dWxr", "dWxz", "dWxn" [H, V]; "dWhr", "dWhz", "dWhn" [H, H];
                "dbxr", "dbxz", "dbxn", "dbhr", "dbhz", "dbhn" [H]; "dx" [B, T, V];
                "dh_prev" / "dh_next" [B, H] (gradient w.r.t. the initial hidden state).
        """
        B, T, H = doutput.shape
        dx: np.ndarray = np.zeros_like(self.cache["x_t"], dtype=np.float32)
        dWxr: np.ndarray = np.zeros_like(self.Wxr, dtype=np.float32)
        dWxz: np.ndarray = np.zeros_like(self.Wxz, dtype=np.float32)
        dWxn: np.ndarray = np.zeros_like(self.Wxn, dtype=np.float32)
        dWhr: np.ndarray = np.zeros_like(self.Whr, dtype=np.float32)
        dWhz: np.ndarray = np.zeros_like(self.Whz, dtype=np.float32)
        dWhn: np.ndarray = np.zeros_like(self.Whn, dtype=np.float32)
        dbxr: np.ndarray = np.zeros_like(self.bxr, dtype=np.float32)
        dbxz: np.ndarray = np.zeros_like(self.bxz, dtype=np.float32)
        dbxn: np.ndarray = np.zeros_like(self.bxn, dtype=np.float32)
        dbhr: np.ndarray = np.zeros_like(self.bhr, dtype=np.float32)
        dbhz: np.ndarray = np.zeros_like(self.bhz, dtype=np.float32)
        dbhn: np.ndarray = np.zeros_like(self.bhn, dtype=np.float32)

        dh_next: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)
        for t in reversed(range(T)):
            x_t = self.cache["x_t"][:, t, :]
            h_prev = self.cache["h_prev"][:, t, :]
            r_t = self.cache["r_t"][:, t, :]
            z_t = self.cache["z_t"][:, t, :]
            n_t = self.cache["n_t"][:, t, :]
            thn_t = self.cache["thn_t"][:, t, :]

            dh = doutput[:, t, :] + dh_next
            # h_t = (1-z_t) * n_t + z_t * h_prev
            # dloss / dz_t =dh *(-n_t + h_prev )
            dz_t = dh * (-n_t + h_prev)
            # dloss / dh * dh / dn_t =  dh * (1-z_t)
            dn_t = dh * (1 - z_t)
            # n_t = np.tanh(an_t)
            # dn_t / dan_t = 1-np.tanh(an_t) **2
            # dloss /dn_t * dn_t * dan_t = dh * (1-z_t) * (1-n_t**2)
            dan_t = dn_t * (1 - n_t**2)
            # an_t = txn + r * thn
            # dan_t / dr_t = thn
            # dloss / dan_t * dan_t / dr_t = thn_t
            dr_t = dan_t * thn_t
            # dan_t / dthn_t = r_t
            # dloss / dan_t * dan_t / dthn_t = dan_t * r_t
            dthn_t = dan_t * r_t
            # r_t = sigmoid(ar_t)
            # dr_t / dar_t = r_t * (1-r_t)
            # dloss / dr_t * dr_t / dar_t = dr_t * r_t * (1-r_t)
            dar_t = dr_t * r_t * (1 - r_t)
            # z_t = sigmoid(az_t)
            # dz_t / daz_t = z_t * (1-z_t)
            daz_t = dz_t * z_t * (1 - z_t)

            # ar_t = x_t @ self.Wxr.T + bxr + h_prev @ self.Whr.T + bhr
            dWxr += dar_t.T @ x_t
            dWhr += dar_t.T @ h_prev
            dbxr += np.sum(dar_t, axis=0)
            dbhr += np.sum(dar_t, axis=0)

            # az_t = x_t @ self.Wxz.T + bxz + h_prev @ self.Whz.T + bhz
            dWxz += daz_t.T @ x_t
            dWhz += daz_t.T @ h_prev
            dbxz += np.sum(daz_t, axis=0)
            dbhz += np.sum(daz_t, axis=0)

            # an_t = x_t @ self.Wxn.T + bxz + r * thn_t
            # thn_t = h_prev @ self.Whn.T + bhn
            dWxn += dan_t.T @ x_t
            dbxn += np.sum(dan_t, axis=0)
            dWhn += dthn_t.T @ h_prev
            dbhn += np.sum(dthn_t, axis=0)

            dx[:, t, :] += dar_t @ self.Wxr + daz_t @ self.Wxz + dan_t @ self.Wxn
            dh_prev = dar_t @ self.Whr + daz_t @ self.Whz + dthn_t @ self.Whn + dh * z_t
            dh_next = dh_prev

        self.diff["dWxr"] = dWxr
        self.diff["dWxz"] = dWxz
        self.diff["dWxn"] = dWxn
        self.diff["dWhr"] = dWhr
        self.diff["dWhz"] = dWhz
        self.diff["dWhn"] = dWhn
        self.diff["dbxr"] = dbxr
        self.diff["dbxz"] = dbxz
        self.diff["dbxn"] = dbxn
        self.diff["dbhr"] = dbhr
        self.diff["dbhz"] = dbhz
        self.diff["dbhn"] = dbhn
        self.diff["dx"] = dx
        self.diff["dh_prev"] = dh_prev
        self.diff["dh_next"] = dh_next

        return self.diff

    def step(
        self, learning_rate: float, diff: dict[str, np.ndarray] | None = None
    ) -> None:
        """
        Apply one in-place SGD update to the 12 GRU weights (not the classifier).

        Args:
            learning_rate (float): SGD learning rate.
            diff (dict[str, np.ndarray] | None): Gradients from ``backward``.
                Defaults to ``self.diff``.
        """
        # set default value
        if diff is None:
            diff = self.diff
        self.Wxr -= learning_rate * diff["dWxr"]
        self.Whr -= learning_rate * diff["dWhr"]
        self.Wxz -= learning_rate * diff["dWxz"]
        self.Whz -= learning_rate * diff["dWhz"]
        self.Wxn -= learning_rate * diff["dWxn"]
        self.Whn -= learning_rate * diff["dWhn"]
        self.bxr -= learning_rate * diff["dbxr"]
        self.bhr -= learning_rate * diff["dbhr"]
        self.bxz -= learning_rate * diff["dbxz"]
        self.bhz -= learning_rate * diff["dbhz"]
        self.bxn -= learning_rate * diff["dbxn"]
        self.bhn -= learning_rate * diff["dbhn"]

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        n_epochs: int = 1000,
        lr: float = 0.01,
        output_name: str = "numpy_gru",
    ) -> None:
        """
        Train with plain SGD and save the weights with the lowest validation loss.

        Same loop as ``RNN.train``: forward -> fc -> softmax -> cross-entropy
        (padding = -1 ignored) -> fc backward and update -> GRU backward -> GRU step.

        Note: this shadows ``nn.Module.train(mode)``; the torch train/eval mode is
        not used here.

        Args:
            train_loader (DataLoader): Training batches of (X [B, T, V], y [B, T]).
            val_loader (DataLoader | None): Validation batches. If None, only the
                train loss is logged and no checkpoint is written.
            n_epochs (int): Number of epochs.
            lr (float): SGD learning rate.
            output_name (str): Checkpoint path without the ".npz" extension.

        The checkpoint "{output_name}.npz" holds the 14 arrays named in ``PARAMS``
        plus "epoch", "val_loss", "lr" and "hidden_size".
        """
        data: dict[str, np.ndarray | None] = {
            "weight_fc": None,
            "bias_fc": None,
            "Wxr": None,
            "Whr": None,
            "bxr": None,
            "bhr": None,
            "Wxz": None,
            "Whz": None,
            "bxz": None,
            "bhz": None,
            "Wxn": None,
            "Whn": None,
            "bxn": None,
            "bhn": None,
        }
        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch in range(n_epochs):
            total_loss: float = 0.0
            count: int = 0
            for X_train, y_train in train_loader:
                X_train, y_train = (
                    X_train.detach().cpu().numpy(),
                    y_train.detach().cpu().numpy(),
                )
                X_train = np.array(X_train, dtype=np.float32)
                y_train = np.array(y_train, dtype=np.int64)
                Batch, Time, _ = X_train.shape
                Hidden = self.hidden_size
                output, hidden = self.forward(X_train)  # [B, T, H]
                out = output @ self.weight_fc.T + self.bias_fc  # [B, T, V]
                logits = out.reshape(-1, self.input_size)  # [B*T, V]
                yhat = softmax(logits)
                y_true = y_train.reshape(-1)  # [B*T]
                loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                output = output.reshape(-1, Hidden)  # [B*T, H]
                dweight_fc = dlogits.T @ output  # [V, H]
                dbias_fc = np.sum(dlogits, axis=0)  # [V]

                # must use weight_fc before it is updated
                doutput = dlogits @ self.weight_fc  # [B*T, H]
                self.weight_fc -= lr * dweight_fc
                self.bias_fc -= lr * dbias_fc

                doutput = doutput.reshape(Batch, Time, Hidden)  # [B, T, H]
                self.backward(doutput)
                self.step(learning_rate=lr)

                n_valid = y_true != -1
                total_loss += loss * n_valid.sum()
                count += n_valid.sum()

            data["weight_fc"] = self.weight_fc.copy()
            data["bias_fc"] = self.bias_fc.copy()
            data["Wxr"] = self.Wxr.copy()
            data["Whr"] = self.Whr.copy()
            data["bxr"] = self.bxr.copy()
            data["bhr"] = self.bhr.copy()
            data["Wxz"] = self.Wxz.copy()
            data["Whz"] = self.Whz.copy()
            data["bxz"] = self.bxz.copy()
            data["bhz"] = self.bhz.copy()
            data["Wxn"] = self.Wxn.copy()
            data["Whn"] = self.Whn.copy()
            data["bxn"] = self.bxn.copy()
            data["bhn"] = self.bhn.copy()

            train_loss = total_loss / count
            train_losses.append(train_loss)
            if val_loader is not None:
                total_loss = 0.0
                count = 0
                for X_val, y_val in val_loader:
                    X_val, y_val = (
                        X_val.detach().cpu().numpy(),
                        y_val.detach().cpu().numpy(),
                    )
                    X_val = np.array(X_val, dtype=np.float32)
                    y_val = np.array(y_val, dtype=np.int64)
                    output, hidden = self.forward(X_val)
                    out = output @ self.weight_fc.T + self.bias_fc
                    logits = out.reshape(-1, self.input_size)
                    yhat = softmax(logits)
                    y_true = y_val.reshape(-1)
                    loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                    n_valid = y_true != -1
                    total_loss += loss * n_valid.sum()
                    count += n_valid.sum()
                val_loss = total_loss / count
                val_losses.append(val_loss)

                if val_loss <= min(val_losses):
                    np.savez(
                        f"{output_name}.npz",
                        **{p: data[p] for p in self.PARAMS},
                        epoch=epoch,
                        val_loss=val_loss,
                        lr=lr,
                        hidden_size=self.hidden_size,
                    )

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"{epoch + 1} has train loss: {train_loss: .4f} validation loss: {val_loss: .4f}"
                    )

            else:
                logger.info(f"{epoch + 1} has train loss: {train_loss: .4f}")

        if val_loader is None:
            logger.info(
                "If you do not have any validation loader, we won't save the npz file any. REMIND this"
            )


class LSTM(nn.Module):
    """
    LSTM layer plus a linear classifier, implemented in NumPy.

    Gate order and weight shapes follow ``torch.nn.LSTM`` (chunk order i, f, g, o)::

        i_t = sigmoid(x_t Wxi^T + bxi + h_{t-1} Whi^T + bhi)
        f_t = sigmoid(x_t Wxf^T + bxf + h_{t-1} Whf^T + bhf)
        g_t = tanh(x_t Wxg^T + bxg + h_{t-1} Whg^T + bhg)
        o_t = sigmoid(x_t Wxo^T + bxo + h_{t-1} Who^T + bho)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    ``forward`` / ``backward`` / ``step`` cover the recurrent layer only. The classifier
    (fc), softmax, cross-entropy and the fc update are done inside ``train``.
    Unlike ``RNN`` / ``GRU``, the torch layers used for initialization are not kept
    as attributes.

    Attributes:
        Wxi, Wxf, Wxg, Wxo (np.ndarray): Input-to-hidden weights, shape [H, V].
        Whi, Whf, Whg, Who (np.ndarray): Hidden-to-hidden weights, shape [H, H].
        bxi, bxf, bxg, bxo, bhi, bhf, bhg, bho (np.ndarray): Biases, shape [H].
        weight_fc (np.ndarray): Classifier weight, shape [V, H].
        bias_fc (np.ndarray): Classifier bias, shape [V].
        cache (dict[str, np.ndarray]): Per-time-step values saved by ``forward`` for
            ``backward``. Keys: "x_t" [B, T, V], "h_prev", "c_prev", "h_t", "c_t",
            "i_t", "f_t", "g_t", "o_t" [B, T, H].
        diff (dict[str, np.ndarray]): Gradients from the latest ``backward``.
        PARAMS (list[str]): Names of the trainable arrays saved to the npz checkpoint.
    """

    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ) -> None:
        """
        Args:
            input_size (int): Vocabulary size V (one-hot input size and number of
                output classes).
            hidden_size (int): Hidden state size H.
            batch_first (bool): If True, inputs are shaped [B, T, V].
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first

        lstm: nn.LSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        lstm_state_dict: dict[str, torch.Tensor] = lstm.state_dict()
        # logger.info(lstm_state_dict.keys())
        Wxi, Wxf, Wxg, Wxo = lstm_state_dict["weight_ih_l0"].chunk(4, 0)
        Whi, Whf, Whg, Who = lstm_state_dict["weight_hh_l0"].chunk(4, 0)
        bxi, bxf, bxg, bxo = lstm_state_dict["bias_ih_l0"].chunk(4, 0)
        bhi, bhf, bhg, bho = lstm_state_dict["bias_hh_l0"].chunk(4, 0)

        self.Wxi, self.Wxf, self.Wxg, self.Wxo = (
            Wxi.detach().cpu().numpy(),
            Wxf.detach().cpu().numpy(),
            Wxg.detach().cpu().numpy(),
            Wxo.detach().cpu().numpy(),
        )
        self.Whi, self.Whf, self.Whg, self.Who = (
            Whi.detach().cpu().numpy(),
            Whf.detach().cpu().numpy(),
            Whg.detach().cpu().numpy(),
            Who.detach().cpu().numpy(),
        )
        self.bxi, self.bxf, self.bxg, self.bxo = (
            bxi.detach().cpu().numpy(),
            bxf.detach().cpu().numpy(),
            bxg.detach().cpu().numpy(),
            bxo.detach().cpu().numpy(),
        )
        self.bhi, self.bhf, self.bhg, self.bho = (
            bhi.detach().cpu().numpy(),
            bhf.detach().cpu().numpy(),
            bhg.detach().cpu().numpy(),
            bho.detach().cpu().numpy(),
        )

        classifier: nn.Linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.input_size
        )
        state_dict: dict[str, torch.Tensor] = classifier.state_dict()
        self.weight_fc: np.ndarray = state_dict["weight"].detach().cpu().numpy()
        self.bias_fc: np.ndarray = state_dict["bias"].detach().cpu().numpy()
        self.hidden: np.ndarray | None = None
        self.cell: np.ndarray | None = None
        self.cache: dict[str, np.ndarray] = {}
        self.diff: dict[str, np.ndarray] = {}
        self.PARAMS: list[str] = [
            "Wxi",
            "Wxf",
            "Wxg",
            "Wxo",
            "Whi",
            "Whf",
            "Whg",
            "Who",
            "bxi",
            "bxf",
            "bxg",
            "bxo",
            "bhi",
            "bhf",
            "bhg",
            "bho",
            "weight_fc",
            "bias_fc",
        ]

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Element-wise logistic sigmoid.

        Args:
            Z (np.ndarray): Pre-activation, any shape.

        Returns:
            np.ndarray: 1 / (1 + exp(-Z)), same shape as Z.
        """
        return 1 / (1 + np.exp(-Z))

    def input_gate(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Input gate i_t.

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            h (np.ndarray): Previous hidden state, shape [B, H].

        Returns:
            np.ndarray: i_t, shape [B, H].
        """
        txi: np.ndarray = X @ self.Wxi.T + self.bxi
        thi: np.ndarray = h @ self.Whi.T + self.bhi
        i_t: np.ndarray = self.sigmoid(txi + thi)
        return i_t

    def forget_gate(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Forget gate f_t.

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            h (np.ndarray): Previous hidden state, shape [B, H].

        Returns:
            np.ndarray: f_t, shape [B, H].
        """
        txf: np.ndarray = X @ self.Wxf.T + self.bxf
        thf: np.ndarray = h @ self.Whf.T + self.bhf
        f_t: np.ndarray = self.sigmoid(txf + thf)
        return f_t

    def output_gate(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Output gate o_t.

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            h (np.ndarray): Previous hidden state, shape [B, H].

        Returns:
            np.ndarray: o_t, shape [B, H].
        """
        txo: np.ndarray = X @ self.Wxo.T + self.bxo
        tho: np.ndarray = h @ self.Who.T + self.bho
        o_t: np.ndarray = self.sigmoid(txo + tho)
        return o_t

    def candidate_g(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Cell candidate g_t (tanh, not sigmoid).

        Args:
            X (np.ndarray): Input at one time step, shape [B, V].
            h (np.ndarray): Previous hidden state, shape [B, H].

        Returns:
            np.ndarray: g_t, shape [B, H].
        """
        txg: np.ndarray = X @ self.Wxg.T + self.bxg
        thg: np.ndarray = h @ self.Whg.T + self.bhg
        g_t: np.ndarray = np.tanh(txg + thg)
        return g_t

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Run the LSTM over the whole sequence, starting from zero hidden and cell states.

        Args:
            X (np.ndarray): One-hot input, shape [B, T, V].

        Returns:
            tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]: (output, (h_last, c_last))
                like torch, where output holds the hidden state of every time step,
                shape [B, T, H], and h_last / c_last are the hidden and cell states of
                the last time step, shape [B, H].
        """
        # return output, (h, c) just like torch
        B, T, V = X.shape
        H = self.hidden_size
        self.cache = {
            "x_t": np.zeros(shape=(B, T, V), dtype=np.float32),
            "h_prev": np.zeros(shape=(B, T, H), dtype=np.float32),
            "c_prev": np.zeros(shape=(B, T, H), dtype=np.float32),
            "h_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "c_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "i_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "f_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "g_t": np.zeros(shape=(B, T, H), dtype=np.float32),
            "o_t": np.zeros(shape=(B, T, H), dtype=np.float32),
        }
        h_prev: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)
        c_prev: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]
            i_t = self.input_gate(X=x_t, h=h_prev)
            f_t = self.forget_gate(X=x_t, h=h_prev)
            g_t = self.candidate_g(X=x_t, h=h_prev)
            o_t = self.output_gate(X=x_t, h=h_prev)

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * np.tanh(c_t)
            self.cache["x_t"][:, t, :] = x_t
            self.cache["h_t"][:, t, :] = h_t
            self.cache["c_t"][:, t, :] = c_t
            self.cache["i_t"][:, t, :] = i_t
            self.cache["f_t"][:, t, :] = f_t
            self.cache["g_t"][:, t, :] = g_t
            self.cache["o_t"][:, t, :] = o_t
            self.cache["h_prev"][:, t, :] = h_prev
            self.cache["c_prev"][:, t, :] = c_prev

            c_prev = c_t
            h_prev = h_t

        # output and (last hidden and last cell)
        return self.cache["h_t"], (h_t, c_t)

    def backward(self, doutput: np.ndarray) -> dict[str, np.ndarray]:
        """
        Backpropagation through time for the LSTM layer, using the latest ``forward`` cache.

        Gradients of the shared weights are summed over all time steps. Two signals
        flow backward in time: dh (dh_prev collects four paths, through i_t, f_t,
        g_t and o_t) and dc (dc = dh * o_t * (1 - tanh(c_t)^2) + dc_next,
        dc_prev = dc * f_t).

        Args:
            doutput (np.ndarray): dLoss/d(output), the gradient flowing into every
                hidden state from the classifier, shape [B, T, H].

        Returns:
            dict[str, np.ndarray]: Gradients (also stored in ``self.diff``):
                "dWxi", "dWxf", "dWxg", "dWxo" [H, V];
                "dWhi", "dWhf", "dWhg", "dWho" [H, H];
                "dbxi", "dbxf", "dbxg", "dbxo", "dbhi", "dbhf", "dbhg", "dbho" [H];
                "dx" [B, T, V]; "dh_prev", "dh_next", "dc_prev", "dc_next" [B, H]
                (gradients w.r.t. the zero initial states after the loop).
        """
        B, T, H = doutput.shape  # [B, T, H]
        dx = np.zeros_like(self.cache["x_t"], dtype=np.float32)  # [B, T, V]
        dWxi = np.zeros_like(self.Wxi, dtype=np.float32)
        dWxf = np.zeros_like(self.Wxf, dtype=np.float32)
        dWxg = np.zeros_like(self.Wxg, dtype=np.float32)
        dWxo = np.zeros_like(self.Wxo, dtype=np.float32)
        dWhi = np.zeros_like(self.Whi, dtype=np.float32)
        dWhf = np.zeros_like(self.Whf, dtype=np.float32)
        dWhg = np.zeros_like(self.Whg, dtype=np.float32)
        dWho = np.zeros_like(self.Who, dtype=np.float32)
        dbxi = np.zeros_like(self.bxi, dtype=np.float32)
        dbxf = np.zeros_like(self.bxf, dtype=np.float32)
        dbxg = np.zeros_like(self.bxg, dtype=np.float32)
        dbxo = np.zeros_like(self.bxo, dtype=np.float32)
        dbhi = np.zeros_like(self.bhi, dtype=np.float32)
        dbhf = np.zeros_like(self.bhf, dtype=np.float32)
        dbhg = np.zeros_like(self.bhg, dtype=np.float32)
        dbho = np.zeros_like(self.bho, dtype=np.float32)

        dh_next: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)
        dc_next: np.ndarray = np.zeros(shape=(B, H), dtype=np.float32)

        for t in reversed(range(T)):
            dh = doutput[:, t, :] + dh_next  # [B, H]
            x_t = self.cache["x_t"][:, t, :]  # [B, V]
            c_t = self.cache["c_t"][:, t, :]  # [B, H]
            h_prev = self.cache["h_prev"][:, t, :]
            c_prev = self.cache["c_prev"][:, t, :]
            i_t = self.cache["i_t"][:, t, :]
            f_t = self.cache["f_t"][:, t, :]
            g_t = self.cache["g_t"][:, t, :]
            o_t = self.cache["o_t"][:, t, :]

            # c_t = f_t * c_prev + i_t * g_t
            # h_t = o_t * tanh(c_t)
            # dloss/do_t = dloss / dh * dh / do_t = tanh(c_t)
            do_t = dh * np.tanh(c_t)
            # dloss /dc_t = dloss / dh * dh / dc_t = dh * o_t * (1-np.tanh(c_t) **2)
            dc = dh * o_t * (1 - np.tanh(c_t) ** 2) + dc_next
            # dloss / df_t = dloss / dc_t * dc_t /df_t = dc * c_prev
            df_t = dc * c_prev
            # dloss / di_t = dc * g_t
            di_t = dc * g_t
            dg_t = dc * i_t

            # i_t, f_t, g_t, o_t
            # i_t = sigmoid(ai_t)
            # ai_t = x_t @ self.Wxi.T + self.bxi + h_prev @ self.Whi.T + self.bhi
            # dloss/dai_t = dloss/di_t * di_t / dai_t
            # dai_t = sigmoid(ai_t) * (1-sigmoid(ai_t))
            dai_t = di_t * i_t * (1 - i_t)
            daf_t = df_t * f_t * (1 - f_t)
            dao_t = do_t * o_t * (1 - o_t)
            # np.tanh
            dag_t = dg_t * (1 - g_t**2)

            # i_t  (dW [H, V] = dai_t^T [H, B] @ x_t [B, V])
            dWxi += dai_t.T @ x_t
            dWhi += dai_t.T @ h_prev
            dbxi += np.sum(dai_t, axis=0)
            dbhi += np.sum(dai_t, axis=0)

            # f_t
            dWxf += daf_t.T @ x_t
            dWhf += daf_t.T @ h_prev
            dbxf += np.sum(daf_t, axis=0)
            dbhf += np.sum(daf_t, axis=0)

            # g_t
            dWxg += dag_t.T @ x_t
            dWhg += dag_t.T @ h_prev
            dbxg += np.sum(dag_t, axis=0)
            dbhg += np.sum(dag_t, axis=0)

            # o_t
            dWxo += dao_t.T @ x_t
            dWho += dao_t.T @ h_prev
            dbxo += np.sum(dao_t, axis=0)
            dbho += np.sum(dao_t, axis=0)

            dx[:, t, :] += (  # [B, H] @ [H, V] -> [B, V]
                dai_t @ self.Wxi
                + daf_t @ self.Wxf
                + dag_t @ self.Wxg
                + dao_t @ self.Wxo
            )
            dh_prev = (  # [B, H] @ [H, H] -> [B, H]
                dai_t @ self.Whi
                + daf_t @ self.Whf
                + dag_t @ self.Whg
                + dao_t @ self.Who
            )
            dc_prev = dc * f_t

            dh_next = dh_prev
            dc_next = dc_prev

        self.diff["dWxi"] = dWxi
        self.diff["dWxf"] = dWxf
        self.diff["dWxg"] = dWxg
        self.diff["dWxo"] = dWxo

        self.diff["dWhi"] = dWhi
        self.diff["dWhf"] = dWhf
        self.diff["dWhg"] = dWhg
        self.diff["dWho"] = dWho

        self.diff["dbxi"] = dbxi
        self.diff["dbxf"] = dbxf
        self.diff["dbxg"] = dbxg
        self.diff["dbxo"] = dbxo

        self.diff["dbhi"] = dbhi
        self.diff["dbhf"] = dbhf
        self.diff["dbhg"] = dbhg
        self.diff["dbho"] = dbho

        self.diff["dx"] = dx
        self.diff["dh_prev"] = dh_prev
        self.diff["dh_next"] = dh_next
        self.diff["dc_prev"] = dc_prev
        self.diff["dc_next"] = dc_next

        return self.diff

    def step(
        self, learning_rate: float, diff: dict[str, np.ndarray] | None = None
    ) -> None:
        """
        Apply one in-place SGD update to the 16 LSTM weights (not the classifier).

        Args:
            learning_rate (float): SGD learning rate.
            diff (dict[str, np.ndarray] | None): Gradients from ``backward``.
                Defaults to ``self.diff``.
        """
        # set default value -> None then use self value
        if diff is None:
            diff = self.diff
        self.Wxi -= learning_rate * diff["dWxi"]
        self.Whi -= learning_rate * diff["dWhi"]
        self.Wxf -= learning_rate * diff["dWxf"]
        self.Whf -= learning_rate * diff["dWhf"]
        self.Wxg -= learning_rate * diff["dWxg"]
        self.Whg -= learning_rate * diff["dWhg"]
        self.Wxo -= learning_rate * diff["dWxo"]
        self.Who -= learning_rate * diff["dWho"]

        self.bxi -= learning_rate * diff["dbxi"]
        self.bhi -= learning_rate * diff["dbhi"]
        self.bxf -= learning_rate * diff["dbxf"]
        self.bhf -= learning_rate * diff["dbhf"]
        self.bxg -= learning_rate * diff["dbxg"]
        self.bhg -= learning_rate * diff["dbhg"]
        self.bxo -= learning_rate * diff["dbxo"]
        self.bho -= learning_rate * diff["dbho"]

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        n_epochs: int = 1000,
        lr: float = 0.01,
        output_name: str = "numpy_lstm",
    ) -> None:
        """
        Train with plain SGD and save the weights with the lowest validation loss.

        Same loop as ``RNN.train``: forward -> fc -> softmax -> cross-entropy
        (padding = -1 ignored) -> fc backward and update -> LSTM backward -> LSTM step.

        Note: this shadows ``nn.Module.train(mode)``; the torch train/eval mode is
        not used here.

        Args:
            train_loader (DataLoader): Training batches of (X [B, T, V], y [B, T]).
            val_loader (DataLoader | None): Validation batches. If None, only the
                train loss is logged and no checkpoint is written.
            n_epochs (int): Number of epochs.
            lr (float): SGD learning rate.
            output_name (str): Checkpoint path without the ".npz" extension.

        The checkpoint "{output_name}.npz" holds the 18 arrays named in ``PARAMS``
        plus "epoch", "val_loss", "lr" and "hidden_size".
        """
        data: dict[str, np.ndarray | None] = {
            "weight_fc": None,
            "bias_fc": None,
            "Wxi": None,
            "Whi": None,
            "bxi": None,
            "bhi": None,
            "Wxf": None,
            "Whf": None,
            "bxf": None,
            "bhf": None,
            "Wxg": None,
            "Whg": None,
            "bxg": None,
            "bhg": None,
            "Wxo": None,
            "Who": None,
            "bxo": None,
            "bho": None,
        }
        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch in range(n_epochs):
            total_loss: float = 0.0
            count: int = 0
            for X_train, y_train in train_loader:
                X_train, y_train = (
                    X_train.detach().cpu().numpy(),
                    y_train.detach().cpu().numpy(),
                )
                X_train = np.array(X_train, dtype=np.float32)
                y_train = np.array(y_train, dtype=np.int64)
                Batch, Time, _ = X_train.shape
                Hidden = self.hidden_size
                output, (hidden, cell) = self.forward(X_train)  # [B, T, H]
                out = output @ self.weight_fc.T + self.bias_fc  # [B, T, V]
                logits = out.reshape(-1, self.input_size)  # [B*T, V]
                yhat = softmax(logits)
                y_true = y_train.reshape(-1)  # [B*T]
                loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                output = output.reshape(-1, Hidden)  # [B*T, H]
                dweight_fc = dlogits.T @ output  # [V, H]
                dbias_fc = np.sum(dlogits, axis=0)  # [V]

                # must use weight_fc before it is updated
                doutput = dlogits @ self.weight_fc  # [B*T, H]
                self.weight_fc -= lr * dweight_fc
                self.bias_fc -= lr * dbias_fc

                doutput = doutput.reshape(Batch, Time, Hidden)  # [B, T, H]
                self.backward(doutput)
                self.step(learning_rate=lr)

                n_valid = y_true != -1
                total_loss += loss * n_valid.sum()
                count += n_valid.sum()

            data["weight_fc"] = self.weight_fc.copy()
            data["bias_fc"] = self.bias_fc.copy()
            data["Wxi"] = self.Wxi.copy()
            data["Whi"] = self.Whi.copy()
            data["bxi"] = self.bxi.copy()
            data["bhi"] = self.bhi.copy()
            data["Wxf"] = self.Wxf.copy()
            data["Whf"] = self.Whf.copy()
            data["bxf"] = self.bxf.copy()
            data["bhf"] = self.bhf.copy()
            data["Wxg"] = self.Wxg.copy()
            data["Whg"] = self.Whg.copy()
            data["bxg"] = self.bxg.copy()
            data["bhg"] = self.bhg.copy()
            data["Wxo"] = self.Wxo.copy()
            data["Who"] = self.Who.copy()
            data["bxo"] = self.bxo.copy()
            data["bho"] = self.bho.copy()

            train_loss = total_loss / count
            train_losses.append(train_loss)
            if val_loader is not None:
                total_loss = 0.0
                count = 0
                for X_val, y_val in val_loader:
                    X_val, y_val = (
                        X_val.detach().cpu().numpy(),
                        y_val.detach().cpu().numpy(),
                    )
                    X_val = np.array(X_val, dtype=np.float32)
                    y_val = np.array(y_val, dtype=np.int64)
                    output, (hidden, cell) = self.forward(X_val)
                    out = output @ self.weight_fc.T + self.bias_fc
                    logits = out.reshape(-1, self.input_size)
                    yhat = softmax(logits)
                    y_true = y_val.reshape(-1)
                    loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                    n_valid = y_true != -1
                    total_loss += loss * n_valid.sum()
                    count += n_valid.sum()
                val_loss = total_loss / count
                val_losses.append(val_loss)

                if val_loss <= min(val_losses):
                    np.savez(
                        f"{output_name}.npz",
                        **{p: data[p] for p in self.PARAMS},
                        epoch=epoch,
                        val_loss=val_loss,
                        lr=lr,
                        hidden_size=self.hidden_size,
                    )
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"{epoch + 1} has train loss: {train_loss: .4f} validation loss: {val_loss: .4f}"
                    )
            else:
                logger.info(f"{epoch + 1} has train loss: {train_loss: .4f}")

        if val_loader is None:
            logger.info(
                "If you do not have any validation loader, we won't save the npz file any. REMIND this"
            )


def main() -> None:
    """Load names.txt, split it into train/validation sets, and run a short training job."""
    input_size: int = 29
    hidden_size: int = 16

    input_token, output_token = set_input_data()
    X_train, X_val, y_train, y_val = set_train_val_set(
        input_token, output_token, ratio=0.8, random_state=42
    )
    # logger.info(X_train.shape)
    train_loader, val_loader = convert_np_to_torch(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        batch_size=16,
        shuffle=True,
    )
    names: list[str] = ["RNN", "GRU", "LSTM"]
    for name in names:
        logger.info(f"{name} training is now working")
        if name == "RNN":
            model: RNN | GRU | LSTM = RNN(
                input_size=input_size, hidden_size=hidden_size, batch_first=True
            )
            n_epochs: int = 200
            lr: float = 0.3
            model.train(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                lr=lr,
            )
        elif name == "GRU":
            model: GRU = GRU(
                input_size=input_size, hidden_size=hidden_size, batch_first=True
            )
            n_epochs = 200
            lr = 0.3
            model.train(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                lr=lr,
            )
        elif name == "LSTM":
            model: LSTM = LSTM(
                input_size=input_size, hidden_size=hidden_size, batch_first=True
            )
            n_epochs = 200
            lr = 0.3
            model.train(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                lr=lr,
            )


if __name__ == "__main__":
    main()
