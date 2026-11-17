import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

import numpy as np

from .utils import (
    CrossEntropyLoss,
    convert_np_to_torch,
    set_input_data,
    set_train_val_set,
    softmax,
)


class RNN(nn.Module):
    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ):
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
        fc_state_dict = self.classifier.state_dict()
        self.weight_fc: np.ndarray = fc_state_dict["weight"].detach().cpu().numpy()
        self.bias_fc: np.ndarray = fc_state_dict["bias"].detach().cpu().numpy()

        self.hidden: np.ndarray | None = None
        self.cache: dict[str, any] = {}
        self.diff: dict[str, any] = {}
        self.PARAMS: list[str] = [
            "weight_fc",
            "bias_fc",
            "weight_ih",
            "weight_hh",
            "bias_ih",
            "bias_hh",
        ]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def backward(self, doutput: np.ndarray) -> dict[str, any]:
        B, T, H = doutput.shape

        dx: np.ndarray = np.zeros(shape=self.cache["x_t"].shape, dtype=np.float32)
        dweight_ih: np.ndarray = np.zeros(shape=self.weight_ih.shape, dtype=np.float32)
        dweight_hh: np.ndarray = np.zeros(shape=self.weight_hh.shape, dtype=np.float32)
        dbias_ih: np.ndarray = np.zeros(shape=self.bias_ih.shape, dtype=np.float32)
        dbias_hh: np.ndarray = np.zeros(shape=self.bias_hh.shape, dtype=np.float32)

        self.diff: dict[str, any] = {
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

    def step(self, learning_rate: float, diff: dict[str, any] = None):
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
        val_loader: DataLoader,
        n_epochs: int = 1000,
        lr: float = 0.01,
        output_name: str = "numpy_rnn",
    ):
        data: dict[str, np.ndarray] = {
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
                output, hidden = self.forward(X_train)
                out = output @ self.weight_fc.T + self.bias_fc
                logits = out.reshape(-1, self.input_size)
                yhat = softmax(logits)
                y_true = y_train.reshape(-1)
                loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                output = output.reshape(-1, Hidden)
                dweight_fc = dlogits.T @ output
                dbias_fc = np.sum(dlogits, axis=0)

                doutput = dlogits @ self.weight_fc
                self.weight_fc -= lr * dweight_fc
                self.bias_fc -= lr * dbias_fc

                doutput = doutput.reshape(Batch, Time, Hidden)
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
    def __init__(
        self, input_size: int = 29, hidden_size: int = 16, batch_first: bool = True
    ):
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
        state_dict = self.classifier.state_dict()
        self.weight_fc: np.ndarray = state_dict["weight"].detach().cpu().numpy()
        self.bias_fc: np.ndarray = state_dict["bias"].detach().cpu().numpy()

        self.cache: dict[str, any] = {}
        self.diff: dict[str, any] = {}
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
        return 1 / (1 + np.exp(-Z))

    def reset_gate(self, X: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        txr = X @ self.Wxr.T + self.bxr
        thr = hidden @ self.Whr.T + self.bhr
        r = self.sigmoid(txr + thr)
        return r

    def update_gate(self, X: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        txz = X @ self.Wxz.T + self.bxz
        thz = hidden @ self.Whz.T + self.bhz
        z = self.sigmoid(txz + thz)
        return z

    def candidate_n(
        self, X: np.ndarray, hidden: np.ndarray, r: np.ndarray
    ) -> np.ndarray:
        txn = X @ self.Wxn.T + self.bxn
        thn = hidden @ self.Whn.T + self.bhn
        n = np.tanh(txn + r * thn)
        return n, thn

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def backward(self, doutput: np.ndarray) -> dict[str, any]:
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

    def step(self, learning_rate: float, diff: dict[str, any] = None):
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
        val_loader: DataLoader,
        n_epochs: int = 1000,
        lr: float = 0.01,
        output_name: str = "numpy_gru",
    ):
        data: dict[str, np.ndarray] = {
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
                output, hidden = self.forward(X_train)
                out = output @ self.weight_fc.T + self.bias_fc
                logits = out.reshape(-1, self.input_size)
                yhat = softmax(logits)
                y_true = y_train.reshape(-1)
                loss, dlogits = CrossEntropyLoss(yhat, y_true, ignore_index=-1)

                output = output.reshape(-1, Hidden)
                dweight_fc = dlogits.T @ output
                dbias_fc = np.sum(dlogits, axis=0)

                doutput = dlogits @ self.weight_fc
                self.weight_fc -= lr * dweight_fc
                self.bias_fc -= lr * dbias_fc

                doutput = doutput.reshape(Batch, Time, Hidden)
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

                logger.info(
                    f"{epoch + 1} has train loss: {train_loss: .4f} validation loss: {val_loss: .4f}"
                )
            else:
                logger.info(f"{epoch + 1} has train loss: {train_loss: .4f}")

        if val_loader is None:
            logger.info(
                "If you do not have any validation loader, we won't save the npz file any. REMIND this"
            )


def main():
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
    name: str = "GRU"
    if name == "RNN":
        model: RNN = RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        n_epochs: int = 1
        lr: float = 0.01
        model.train(
            train_loader=train_loader, val_loader=val_loader, n_epochs=n_epochs, lr=lr
        )
    elif name == "GRU":
        model: GRU = GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        n_epochs: int = 1
        lr: float = 0.01
        model.train(
            train_loader=train_loader, val_loader=val_loader, n_epochs=n_epochs, lr=lr
        )


if __name__ == "__main__":
    main()
