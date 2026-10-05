"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

import copy
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from torch import nn

from platform_components.EdgeLake_functions.blockchain_EL_functions import fetch_data_from_db
from platform_components.data_handlers.shakespeare_common import (
    CHARSET,
    DEFAULT_ROUND_SECONDS,
    DEFAULT_SEQ_LEN,
    STOI,
    decode_ids,
    pad_ids,
    parse_epoch,
    max_round_query,
    round_bounds,
    select_samples_query,
)
from platform_components.lib.logger.logger_config import configure_logging
from platform_components.lib.modules.local_model_update import LocalModelUpdate
from platform_components.model_fusion_algorithms.FedAvg import FedAvg_aggregate


def _torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NextCharModel(nn.Module):
    """Embedding, LSTM, and a linear head over the next character."""

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        sequence, _state = self.lstm(embedded)
        return self.output(sequence)


LOGICAL_DATABASE = os.getenv("LOGICAL_DATABASE")
TRAIN_TABLE = os.getenv("TRAIN_TABLE")
TEST_TABLE = os.getenv("TEST_TABLE")


class ShakespeareDataHandler:
    """Next-character Shakespeare model. Each round reads rows stamped with that round_number."""

    def __init__(self, node_name):
        configure_logging("node_server_data_handler")
        self.logger = logging.getLogger(__name__)
        self.node_name = node_name
        self.tcp_ip_port = os.getenv("EXTERNAL_TCP_IP_PORT")
        self.edgelake_node_url = f'http://{os.getenv("EXTERNAL_IP")}'
        self.db_name = LOGICAL_DATABASE
        self.seq_len = int(os.getenv("SEQ_LEN", str(DEFAULT_SEQ_LEN)))
        self.device = _torch_device()
        self.fl_model = self.model_def().to(self.device)
        self.logger.info("Shakespeare model on %s", self.device)

    def model_def(self):
        return NextCharModel(vocab_size=len(CHARSET))

    def get_weights(self):
        return [param.detach().cpu().numpy() for param in self.fl_model.parameters()]

    def update_model(self, weights):
        if isinstance(weights, LocalModelUpdate):
            weights = weights.get("weights")
        params = list(self.fl_model.parameters())
        if len(list(weights)) != len(params):
            raise ValueError(
                f"Expected {len(params)} parameter tensors, received {len(list(weights))}"
            )
        with torch.no_grad():
            for param, value in zip(params, weights):
                tensor = torch.as_tensor(np.asarray(value), dtype=param.dtype, device=param.device)
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(f"Shape mismatch for {tuple(param.shape)} vs {tuple(tensor.shape)}")
                param.copy_(tensor)

    def aggregate_model_weights(self, weights):
        return FedAvg_aggregate(weights)

    def _epoch_start(self):
        raw = os.getenv("DATA_EPOCH_START")
        if not raw:
            raise RuntimeError(
                "DATA_EPOCH_START is not set. Use the epoch printed by the Shakespeare publisher."
            )
        return parse_epoch(raw)

    def _round_duration(self):
        return float(os.getenv("ROUND_DURATION_SECONDS", str(DEFAULT_ROUND_SECONDS)))

    def _wait_until_window_closes(self, end):
        grace = float(os.getenv("QUERY_GRACE_SECONDS", "2"))
        max_wait = float(os.getenv("MAX_WAIT_SECONDS", "86400"))
        deadline = end + timedelta(seconds=grace)
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        if remaining > max_wait:
            raise RuntimeError(
                "Round window is still far in the future. Check that DATA_EPOCH_START and "
                "ROUND_DURATION_SECONDS match the publisher."
            )
        self.logger.info(
            "Waiting %.1fs until inserted rows for this round are queryable (through %s)",
            remaining,
            end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        while True:
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5))

    def _row_field(self, row, name):
        if name in row:
            return row[name]
        for key, value in row.items():
            if str(key).lower() == name:
                return value
        raise KeyError(name)

    def _rows_to_arrays(self, payload, query):
        if not isinstance(payload, dict) or not payload.get("Query"):
            raise IOError(f"No rows returned for round query: {query}. Response: {payload}")
        rows = payload["Query"]
        rows = sorted(rows, key=lambda row: str(self._row_field(row, "timestamp")))
        sequences = []
        labels = []
        for row in rows:
            sequence = str(self._row_field(row, "sequence"))
            label = str(self._row_field(row, "label"))
            if len(sequence) != self.seq_len or len(label) != 1:
                raise IOError(
                    f"Expected sequence length {self.seq_len} and a one-character label. "
                    f"Got sequence length {len(sequence)} and label {label!r}. "
                    "SEQ_LEN must match the publisher."
                )
            if any(ch not in STOI for ch in sequence) or label not in STOI:
                raise IOError("Row contains a character outside the Shakespeare vocabulary")
            sequences.append([STOI[ch] for ch in sequence])
            labels.append(STOI[label])
        x_values = np.array(sequences, dtype=np.int32)
        y_values = np.array(labels, dtype=np.int32)
        self.logger.info("Loaded %s labeled sequences", len(y_values))
        return x_values, y_values

    def _round_query(self, table, round_number, history_rounds=1):
        if not self.tcp_ip_port:
            raise RuntimeError("EXTERNAL_TCP_IP_PORT is not set")
        return select_samples_query(self.db_name, table, round_number, history_rounds)

    def _fetch_round(self, table, round_number, history_rounds=1):
        query = self._round_query(table, round_number, history_rounds)
        self.logger.info("Querying %s", query)
        payload = fetch_data_from_db(self.edgelake_node_url, query, self.tcp_ip_port)
        return self._rows_to_arrays(payload, query)

    def load_dataset(self, node_name, round_number):
        _start, end = round_bounds(round_number, self._epoch_start(), self._round_duration())
        self.logger.info("Round %s reads rows where round_number = %s", round_number, round_number)
        self._wait_until_window_closes(end)
        x_train, y_train = self._fetch_round(TRAIN_TABLE, round_number)
        x_test, y_test = self._fetch_round(TEST_TABLE, round_number)
        return (x_train, y_train), (x_test, y_test)

    def _fit(self, x_train, y_train):
        epochs = int(os.getenv("TRAIN_EPOCHS", "15"))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        features = torch.as_tensor(x_train, dtype=torch.long, device=self.device)
        labels = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.fl_model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        best_loss = None
        best_state = None

        for epoch in range(epochs):
            self.fl_model.train()
            order = torch.randperm(features.shape[0], device=self.device)
            total_loss = 0.0
            batches = 0
            for start in range(0, features.shape[0], batch_size):
                batch_index = order[start:start + batch_size]
                optimizer.zero_grad()
                batch_features = features[batch_index]
                logits = self.fl_model(batch_features)
                targets = batch_features.clone()
                targets[:, :-1] = batch_features[:, 1:]
                targets[:, -1] = labels[batch_index]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fl_model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
            epoch_loss = total_loss / max(batches, 1)
            self.logger.info("Epoch %s/%s loss %.4f", epoch + 1, epochs, epoch_loss)
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self.fl_model.state_dict())

        if best_state is not None:
            self.fl_model.load_state_dict(best_state)
        return self.get_weights()

    def train(self, round_number):
        history = int(os.getenv("TRAIN_HISTORY_ROUNDS", "12"))
        x_train, y_train = self._fetch_round(TRAIN_TABLE, round_number, history)
        self.logger.info(
            "Training on %s sequences from the last %s rounds through round %s",
            len(y_train),
            history,
            round_number,
        )
        return self._fit(x_train, y_train)

    def _latest_stored_round(self, table):
        """Highest round_number that has been written, not the wall-clock round."""
        query = max_round_query(self.db_name, table)
        payload = fetch_data_from_db(self.edgelake_node_url, query, self.tcp_ip_port)
        rows = payload.get("Query") if isinstance(payload, dict) else None
        if not rows:
            raise IOError(f"No rows in {table}. Response: {payload}")
        raw = None
        for key, value in rows[0].items():
            if "round" in str(key).lower():
                raw = value
                break
        if raw in (None, ""):
            raise IOError(f"No rows in {table}. Response: {payload}")
        return int(raw)

    def run_inference(self):
        """Character accuracy on the test rows for the newest stored round."""
        round_number = self._latest_stored_round(TEST_TABLE)
        x_test, y_test = self._fetch_round(TEST_TABLE, round_number)
        self.fl_model.eval()
        with torch.no_grad():
            logits = self.fl_model(torch.as_tensor(x_test, dtype=torch.long, device=self.device))[:, -1, :]
        predicted = logits.argmax(dim=-1).detach().cpu().numpy()
        accuracy = float(np.mean(predicted == y_test) * 100.0)
        self.logger.info("Round %s test character accuracy: %.2f", round_number, accuracy)
        return {
            "model_accuracy": round(accuracy, 2),
            "round_number": round_number,
            "samples": int(y_test.shape[0]),
        }

    def _sample_index(self, probabilities):
        temperature = float(os.getenv("INFERENCE_TEMPERATURE", "0.7"))
        top_k = int(os.getenv("INFERENCE_TOP_K", "12"))
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if temperature <= 0 or top_k == 1:
            return int(np.argmax(probabilities))
        if 0 < top_k < len(probabilities):
            keep = np.argpartition(probabilities, -top_k)[-top_k:]
            chosen = np.full(len(probabilities), -np.inf)
            chosen[keep] = np.log(np.clip(probabilities[keep], 1e-12, 1.0))
        else:
            chosen = np.log(np.clip(probabilities, 1e-12, 1.0))
        scaled = chosen / temperature
        scaled -= np.max(scaled)
        weights = np.exp(scaled)
        weights /= np.sum(weights)
        return int(np.random.choice(len(weights), p=weights))

    def direct_inference(self, data):
        """
        Continue a prompt. `data` is a list of Shakespeare vocabulary indexes
        (the same encoding the GUI sends).
        """
        if not data:
            raise ValueError("Prompt is empty")
        prompt_ids = [int(value) for value in data]
        prompt = decode_ids(prompt_ids)
        gen_len = int(os.getenv("GEN_LEN", "40"))
        window = pad_ids(prompt_ids, self.seq_len)
        generated = []
        top_k = []
        self.fl_model.eval()
        with torch.no_grad():
            for step in range(gen_len):
                tokens = torch.tensor([window], dtype=torch.long, device=self.device)
                probabilities = torch.softmax(self.fl_model(tokens)[0, -1], dim=-1).detach().cpu().numpy()
                if step == 0:
                    order = np.argsort(probabilities)[::-1][:5]
                    top_k = [
                        {"char": CHARSET[int(index)], "probability": round(float(probabilities[index]), 4)}
                        for index in order
                    ]
                nxt = self._sample_index(probabilities)
                generated.append(nxt)
                window = window[1:] + [nxt]
        continuation = decode_ids(generated)
        return {
            "prompt": prompt,
            "continuation": continuation,
            "text": prompt + continuation,
            "next_char": continuation[:1],
            "top_k": top_k,
        }
