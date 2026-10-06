"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""
Federated BloodMNIST classifier.

Each operator trains this CNN on the blood-cell slides whose filenames the
publisher inserted for that operator. The PNG stays on disk (IMAGE_ROOT_DIR);
AnyLog stores the filename, size, and class. A round is not a single pass:
TRAIN_EPOCHS walks the node's current rows several times, and TRAIN_HISTORY_ROUNDS
keeps the previous rounds
in the fit so accuracy can climb after a few aggregations. FedAvg averages the
NumPy parameter tensors returned by get_weights. The network has no BatchNorm
because those running statistics are not parameters and would not be averaged.
"""

import copy
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

from platform_components.EdgeLake_functions.blockchain_EL_functions import fetch_data_from_db
from platform_components.data_handlers.medmnist_common import (
    CLASS_NAMES,
    DEFAULT_HISTORY_ROUNDS,
    DEFAULT_TRAIN_EPOCHS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    image_to_model_batch,
    file_stem,
    max_round_query,
    select_image_info_query,
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


class MedMNISTCNN(nn.Module):
    """
    Convolutional classifier for a 28x28 MedMNIST image.

    Capacity is enough for BloodMNIST without being painful to average across
    nodes. Dropout carries no parameters, so it does not affect FedAvg.
    """

    def __init__(self, in_channels=IMAGE_CHANNELS, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        # images: float NCHW in [0, 1]
        return self.classifier(self.features(images))


class MedMNISTDataHandler:
    """Local BloodMNIST trainer. One instance lives on each EdgeFL node."""

    def __init__(self, node_name):
        configure_logging("node_server_data_handler")
        self.logger = logging.getLogger(__name__)
        self.node_name = node_name
        self.tcp_ip_port = os.getenv("EXTERNAL_TCP_IP_PORT")
        self.edgelake_node_url = f'http://{os.getenv("EXTERNAL_IP")}'
        self.db_name = os.getenv("LOGICAL_DATABASE")
        self.train_table = os.getenv("TRAIN_TABLE")
        self.test_table = os.getenv("TEST_TABLE")
        self.device = _torch_device()
        self.fl_model = self.model_def().to(self.device)
        parameter_count = sum(parameter.numel() for parameter in self.fl_model.parameters())
        self.logger.info(
            "MedMNIST CNN on %s (%s parameters, %s classes)",
            self.device,
            parameter_count,
            len(CLASS_NAMES),
        )

    def model_def(self):
        return MedMNISTCNN()

    def get_weights(self):
        return [parameter.detach().cpu().numpy() for parameter in self.fl_model.parameters()]

    def update_model(self, weights):
        if isinstance(weights, LocalModelUpdate):
            weights = weights.get("weights")
        parameters = list(self.fl_model.parameters())
        if len(list(weights)) != len(parameters):
            raise ValueError(
                f"Expected {len(parameters)} parameter tensors, received {len(list(weights))}"
            )
        with torch.no_grad():
            for parameter, value in zip(parameters, weights):
                tensor = torch.as_tensor(np.asarray(value), dtype=parameter.dtype, device=parameter.device)
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"Shape mismatch for {tuple(parameter.shape)} vs {tuple(tensor.shape)}"
                    )
                parameter.copy_(tensor)

    def aggregate_model_weights(self, weights):
        return FedAvg_aggregate(weights)

    def _row_field(self, row, name):
        if name in row:
            return row[name]
        for key, value in row.items():
            if str(key).lower() == name:
                return value
        raise KeyError(name)

    def _image_root(self):
        """Directory of PNGs written by publish_medmnist_info.py."""
        root = os.getenv("IMAGE_ROOT_DIR", "edgefl/data/medmnist/published_images")
        if os.path.isabs(root):
            return root
        github = os.getenv("GITHUB_DIR")
        if not github:
            raise RuntimeError("GITHUB_DIR is not set; cannot locate published MedMNIST slides")
        return os.path.join(github, root)

    def _load_slide(self, row):
        """Open the PNG named by this row. The pixel values are not in AnyLog."""
        filename = str(self._row_field(row, "filename"))
        path = os.path.join(self._image_root(), filename)
        if not os.path.isfile(path):
            raise IOError(f"Published slide is not on disk: {path}")
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
            raise IOError(f"{filename} has shape {array.shape}, expected {(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)}")
        return array

    def _rows_to_arrays(self, payload, query):
        if not isinstance(payload, dict) or not payload.get("Query"):
            raise IOError(f"No rows returned for round query: {query}. Response: {payload}")
        rows = payload["Query"]
        images = []
        labels = []
        for row in rows:
            image = self._load_slide(row)
            label = int(self._row_field(row, "label"))
            if label < 0 or label >= len(CLASS_NAMES):
                raise IOError(f"Label {label} is outside the BloodMNIST class list")
            images.append(np.transpose(image, (2, 0, 1)))
            labels.append(label)
        # NCHW float in [0, 1], the layout MedMNISTCNN.forward expects.
        features = np.stack(images).astype(np.float32) / 255.0
        targets = np.asarray(labels, dtype=np.int64)
        self.logger.info("Loaded %s labeled images", len(targets))
        return features, targets

    def _fetch_round(self, table, round_number, history_rounds):
        """Load rows for this round_number. Retry briefly while the publisher is still inserting."""
        if not self.tcp_ip_port:
            raise RuntimeError("EXTERNAL_TCP_IP_PORT is not set")
        if not self.db_name or not table:
            raise RuntimeError("LOGICAL_DATABASE and the table name must be set")
        query = select_image_info_query(self.db_name, table, round_number, history_rounds)
        retries = int(os.getenv("QUERY_RETRIES", "15"))
        pause = float(os.getenv("QUERY_RETRY_SECONDS", "2"))
        last_error = None
        for attempt in range(1, retries + 1):
            self.logger.info("Querying %s", query)
            payload = fetch_data_from_db(self.edgelake_node_url, query, self.tcp_ip_port)
            try:
                return self._rows_to_arrays(payload, query)
            except IOError as error:
                last_error = error
                self.logger.info(
                    "No rows yet for %s (attempt %s/%s). The publisher may still be flushing.",
                    table,
                    attempt,
                    retries,
                )
                time.sleep(pause)
        raise last_error

    def _augment(self, batch):
        """
        Random flips. Blood cells have no canonical orientation, so this gives
        each local epoch a different view of the same slides.
        """
        batch = batch.clone()
        flip_horizontal = torch.rand(batch.shape[0], device=batch.device) < 0.5
        flip_vertical = torch.rand(batch.shape[0], device=batch.device) < 0.5
        for index in range(batch.shape[0]):
            if flip_horizontal[index]:
                batch[index] = torch.flip(batch[index], dims=[2])
            if flip_vertical[index]:
                batch[index] = torch.flip(batch[index], dims=[1])
        return batch

    def _fit(self, features, targets):
        epochs = int(os.getenv("TRAIN_EPOCHS", str(DEFAULT_TRAIN_EPOCHS)))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        if epochs < 1:
            raise RuntimeError("TRAIN_EPOCHS must be at least 1")
        steps_per_epoch = int(np.ceil(len(targets) / batch_size))
        self.logger.info(
            "Local training on %s images for %s epochs (%s optimizer steps per epoch, batch %s)",
            len(targets),
            epochs,
            steps_per_epoch,
            batch_size,
        )

        images = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.fl_model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        best_loss = None
        best_state = None

        for epoch in range(epochs):
            self.fl_model.train()
            order = torch.randperm(images.shape[0], device=self.device)
            total_loss = 0.0
            correct = 0
            seen = 0
            batches = 0
            for start in range(0, images.shape[0], batch_size):
                batch_index = order[start:start + batch_size]
                batch_images = self._augment(images[batch_index])
                batch_labels = labels[batch_index]
                optimizer.zero_grad()
                logits = self.fl_model(batch_images)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fl_model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                correct += int((logits.argmax(dim=-1) == batch_labels).sum().detach())
                seen += int(batch_labels.shape[0])
                batches += 1
            epoch_loss = total_loss / max(batches, 1)
            epoch_accuracy = 100.0 * correct / max(seen, 1)
            self.logger.info(
                "Epoch %s/%s loss %.4f train-acc %.2f",
                epoch + 1,
                epochs,
                epoch_loss,
                epoch_accuracy,
            )
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self.fl_model.state_dict())

        if best_state is not None:
            self.fl_model.load_state_dict(best_state)
        return self.get_weights()

    def train(self, round_number):
        history = int(os.getenv("TRAIN_HISTORY_ROUNDS", str(DEFAULT_HISTORY_ROUNDS)))
        features, targets = self._fetch_round(self.train_table, round_number, history)
        counts = np.bincount(targets, minlength=len(CLASS_NAMES))
        class_summary = ", ".join(
            f"{name}={int(counts[index])}" for index, name in enumerate(CLASS_NAMES) if counts[index]
        )
        self.logger.info(
            "Round %s training set covers the last %s rounds: %s",
            round_number,
            history,
            class_summary,
        )
        return self._fit(features, targets)

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

    def _scores(self, features, targets):
        probabilities = self._predict(features)
        predicted = probabilities.argmax(axis=1)
        accuracy = float(np.mean(predicted == targets) * 100.0)
        per_class = []
        for class_id, name in enumerate(CLASS_NAMES):
            mask = targets == class_id
            if not np.any(mask):
                continue
            per_class.append({
                "label": class_id,
                "class_name": name,
                "file_stem": file_stem(class_id),
                "accuracy": round(float(np.mean(predicted[mask] == class_id) * 100.0), 2),
                "samples": int(mask.sum()),
            })
        return accuracy, per_class

    def _predict(self, features):
        self.fl_model.eval()
        with torch.no_grad():
            logits = self.fl_model(torch.as_tensor(features, dtype=torch.float32, device=self.device))
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.detach().cpu().numpy()

    def run_inference(self):
        """Accuracy on this node's own test rows, over the recent stored rounds."""
        round_number = self._latest_stored_round(self.test_table)
        history = int(os.getenv("EVAL_HISTORY_ROUNDS", str(DEFAULT_HISTORY_ROUNDS)))
        features, targets = self._fetch_round(self.test_table, round_number, history)
        accuracy, per_class = self._scores(features, targets)
        self.logger.info(
            "Test accuracy through round %s: %.2f on %s images",
            round_number,
            accuracy,
            len(targets),
        )
        return {
            "model_accuracy": round(accuracy, 2),
            "round_number": round_number,
            "samples": int(targets.shape[0]),
            "per_class": per_class,
        }

    def _images_from_request(self, data):
        """
        Accept a flat 28x28x3 RGB list, or one HWC/NHWC matrix.

        Values above 1.5 are treated as 0-255 pixels. Anything else is already
        scaled to [0, 1]. An uploaded image file does not come through here;
        direct_inference decodes that file first.
        """
        array = np.asarray(data, dtype=np.float32)
        expected = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS
        if array.ndim == 1 and array.size == expected:
            array = array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
        elif array.ndim == 3 and array.shape == (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
            array = array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
        elif array.ndim == 4 and tuple(array.shape[1:]) == (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
            pass
        else:
            raise ValueError(
                f"Expected {expected} RGB values or a {IMAGE_SIZE}x{IMAGE_SIZE}x{IMAGE_CHANNELS} image, "
                f"got shape {tuple(array.shape)}"
            )
        if float(np.nanmax(array)) > 1.5:
            array = array / 255.0
        # NHWC -> NCHW
        return np.transpose(array, (0, 3, 1, 2))

    def _uploaded_image(self, data):
        """The image file the user attached, or None when `data` is already a matrix."""
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            text = data.strip()
        elif isinstance(data, (list, tuple)) and len(data) == 1 and isinstance(data[0], str):
            text = data[0].strip()
        else:
            return None
        if text.startswith("data:image/") or text.startswith("iVBOR") or text.startswith("/9j/"):
            return text
        return None

    def direct_inference(self, data, labels=None):
        """
        Classify one image the caller provides.

        A PNG or JPG is converted here to the 28x28x3 matrix the network
        expects. A numeric matrix is used as-is. This does not query AnyLog.
        `labels`, when the aggregator passes them, is an optional expected
        class id and is not required by the GUI.
        """
        uploaded = self._uploaded_image(data)
        features = image_to_model_batch(uploaded) if uploaded is not None else self._images_from_request(data)
        probabilities = self._predict(features)[0]
        label = int(np.argmax(probabilities))
        ranking = np.argsort(probabilities)[::-1]
        result = {
            "label": label,
            "class_name": CLASS_NAMES[label],
            "file_stem": file_stem(label),
            "confidence": round(float(probabilities[label]), 4),
            "probabilities": [
                {
                    "label": int(class_id),
                    "class_name": CLASS_NAMES[int(class_id)],
                    "file_stem": file_stem(int(class_id)),
                    "probability": round(float(probabilities[int(class_id)]), 4),
                }
                for class_id in ranking
            ],
        }
        if labels not in (None, "", []):
            expected = int(labels[0] if isinstance(labels, (list, tuple)) else labels)
            result["expected_label"] = expected
            result["correct"] = expected == label
        return result
