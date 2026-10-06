"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""Download BloodMNIST and return the arrays the publisher streams."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_components.data_handlers.medmnist_common import (  # noqa: E402
    NPZ_MD5,
    NPZ_URL,
)

NPZ_PATH = Path(__file__).resolve().parent / "bloodmnist.npz"


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_npz(path=NPZ_PATH, url=NPZ_URL, md5=NPZ_MD5):
    """Download the MedMNIST npz once and refuse a file whose checksum does not match."""
    path = Path(path)
    if path.is_file() and _md5(path) == md5:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BloodMNIST to {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    digest = _md5(temporary)
    if digest != md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"BloodMNIST checksum {digest} did not match {md5}")
    temporary.replace(path)
    return path


def load_splits(path=NPZ_PATH):
    """
    Train, validation, and test arrays.

    Validation slides are kept separate so the caller can fold them into the
    training stream. That gives each operator more local images than the
    official train split alone. Test stays held out.
    """
    path = ensure_npz(path)
    with np.load(path) as data:
        splits = {
            "train_images": np.array(data["train_images"]),
            "train_labels": np.array(data["train_labels"]).reshape(-1),
            "val_images": np.array(data["val_images"]),
            "val_labels": np.array(data["val_labels"]).reshape(-1),
            "test_images": np.array(data["test_images"]),
            "test_labels": np.array(data["test_labels"]).reshape(-1),
        }
    return splits


def load_train_and_test(path=NPZ_PATH):
    """Training images (official train + val) and the held-out test split."""
    splits = load_splits(path)
    images = np.concatenate([splits["train_images"], splits["val_images"]], axis=0)
    labels = np.concatenate([splits["train_labels"], splits["val_labels"]], axis=0)
    return images, labels, splits["test_images"], splits["test_labels"]
