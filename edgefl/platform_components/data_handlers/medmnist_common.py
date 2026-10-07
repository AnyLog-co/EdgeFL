"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""Shared BloodMNIST constants for the publisher and the training nodes.

The demo uses BloodMNIST, the 8-class blood-cell subset of MedMNIST v2.
Images are 28x28 RGB. Training and test rows store that matrix as a list of
numbers plus the class label. They do not store an image file or a blob.
A user upload is converted to the same matrix inside direct_inference.
Eval-image filenames are built from CLASS_NAMES so a file name is the
correct prediction.
"""

import ast
import base64
import io
import json
from datetime import datetime, timezone

import numpy as np


# MedMNIST label id -> name shown by the GUI and used in eval filenames.
# Label 3's official MedMNIST string is
# "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)".
# The short name is what a prediction is compared with on disk.
CLASS_NAMES = (
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
)

# One held-out test image per entry. Six slides, not all eight classes,
# so the eval folder stays small enough to click through.
EVAL_LABELS = (0, 1, 2, 4, 6, 7)

IMAGE_SIZE = 28
IMAGE_CHANNELS = 3
# Eval PNGs are nearest-neighbor scaled by this factor so a person can see
# the cell. 28 * 8 = 224. The GUI averages those blocks back to 28x28.
DISPLAY_SCALE = 8

# Defaults shared by the publisher and the env files.
DEFAULT_SAMPLES_PER_ROUND = 1536
DEFAULT_TEST_SAMPLES_PER_ROUND = 192
DEFAULT_TRAIN_EPOCHS = 20
DEFAULT_HISTORY_ROUNDS = 4

DATASET_NAME = "bloodmnist"
NPZ_URL = "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1"
NPZ_MD5 = "7053d0359d879ad8a5505303e11de1dc"


def file_stem(label):
    """`6_neutrophil` — the stem of the eval file whose correct class is `label`."""
    label = int(label)
    slug = CLASS_NAMES[label].replace(" ", "_")
    return f"{label}_{slug}"


def parse_epoch(value):
    """Parse a publisher epoch as UTC."""
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip().strip('"').strip("'")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_ts(moment):
    """Fixed-width UTC timestamp stored on each row."""
    moment = parse_epoch(moment).replace(tzinfo=None)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")


def portion_counts(total, parts):
    """Split `total` items across `parts` buckets. Counts differ by at most one."""
    total = int(total)
    parts = int(parts)
    if parts < 1:
        raise ValueError("at least one bucket is required")
    if total < parts:
        raise ValueError(f"Need at least one item for each of {parts} buckets, got {total}")
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def select_image_info_query(db_name, table, round_number, history_rounds=1):
    """
    Scalar image info for this round, plus earlier rounds when history_rounds > 1.

    Same shape as the chest X-ray publisher: filename, size, and class.
    The pixel matrix stays in the PNG named by filename.
    """
    round_number = int(round_number)
    history_rounds = max(1, int(history_rounds))
    first_round = max(1, round_number - history_rounds + 1)
    if first_round == round_number:
        where = f"round_number = {round_number}"
    else:
        where = f"round_number >= {first_round} AND round_number <= {round_number}"
    return (
        f"sql {db_name} format=json and stat=false "
        f"SELECT timestamp, filename, width, height, label, class_name, round_number "
        f"FROM {table} WHERE {where}"
    )


def select_images_query(db_name, table, round_number, history_rounds=1):
    """Rows inserted for this round, plus earlier rounds when history_rounds > 1."""
    round_number = int(round_number)
    history_rounds = max(1, int(history_rounds))
    first_round = max(1, round_number - history_rounds + 1)
    if first_round == round_number:
        where = f"round_number = {round_number}"
    else:
        where = f"round_number >= {first_round} AND round_number <= {round_number}"
    return (
        f"sql {db_name} format=json and stat=false "
        f"SELECT timestamp, matrix, label, round_number FROM {table} "
        f"WHERE {where}"
    )


def max_round_query(db_name, table):
    return (
        f"sql {db_name} format=json and stat=false "
        f"SELECT max(round_number) FROM {table}"
    )


def encode_matrix(image):
    """
    Flat list of 28x28x3 pixel values, row-major, channels last.

    The publisher puts this list on the row as `matrix`. json.dumps then sends
    a numeric array, which AnyLog stores as numbers. A JSON string of the same
    numbers would be stored as a blob, so this must stay a list.
    """
    array = np.ascontiguousarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
        raise ValueError(
            f"Expected image shape {(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)}, got {array.shape}"
        )
    return [int(value) for value in array.reshape(-1)]


def decode_matrix(payload):
    """Parse a stored matrix into uint8 HWC. Accepts a list or the text AnyLog returns."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        text = payload.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # AnyLog may hand the matrix back as a Python-style literal.
            payload = ast.literal_eval(text)
    array = np.asarray(payload, dtype=np.uint8)
    expected = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    if array.shape != expected:
        array = array.reshape(expected)
    return np.ascontiguousarray(array)


def _rgb_from_upload(payload):
    """PNG or JPG bytes, a base64 string, or a data URL -> uint8 HWC RGB."""
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[1]
        raw = base64.b64decode(text)
    else:
        raw = bytes(payload)
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_to_model_batch(payload):
    """
    User-uploaded image -> one float32 NCHW example in [0, 1].

    This is the conversion direct_inference runs. The publisher never sees
    the file.
    """
    averaged = box_downsample(_rgb_from_upload(payload), IMAGE_SIZE)
    if averaged.shape != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
        raise ValueError(
            f"Expected {(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)} after resize, got {averaged.shape}"
        )
    return np.transpose(averaged, (2, 0, 1))[None, ...].astype(np.float32)


def upscale_nearest(image, factor=DISPLAY_SCALE):
    """Repeat each pixel into a factor x factor block. Used for the eval PNGs."""
    factor = int(factor)
    if factor < 1:
        raise ValueError("scale factor must be positive")
    array = np.asarray(image)
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def box_downsample(image, size=IMAGE_SIZE):
    """
    Average an HWC uint8 image into `size` x `size`.

    Block edges are floor(i * length / size). A nearest-neighbor upscale by an
    integer factor round-trips through this exactly, so the labeled eval PNGs
    classify as themselves. direct_inference uses the same average.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, channels = array.shape
    output = np.empty((size, size, channels), dtype=np.float32)
    for y in range(size):
        y0 = (y * height) // size
        y1 = max(y0 + 1, ((y + 1) * height) // size)
        for x in range(size):
            x0 = (x * width) // size
            x1 = max(x0 + 1, ((x + 1) * width) // size)
            block = array[y0:y1, x0:x1].astype(np.float32)
            output[y, x] = block.reshape(-1, channels).mean(axis=0) / 255.0
    return output


def select_eval_indices(labels):
    """
    First test-set row of each EVAL_LABELS class.

    The exporter and the publisher both call this, so those slides are written
    to eval_images/ and are not inserted into the test table.
    """
    flat = np.asarray(labels).reshape(-1)
    chosen = {}
    for index, label in enumerate(flat.tolist()):
        label = int(label)
        if label in EVAL_LABELS and label not in chosen:
            chosen[label] = index
        if len(chosen) == len(EVAL_LABELS):
            break
    missing = [label for label in EVAL_LABELS if label not in chosen]
    if missing:
        names = ", ".join(CLASS_NAMES[label] for label in missing)
        raise RuntimeError(f"Test split is missing eval classes: {names}")
    return chosen


def _self_check():
    assert format_ts("2026-10-01T17:00:00Z") == "2026-10-01 17:00:00.000000"
    assert file_stem(6) == "6_neutrophil"
    assert portion_counts(1536, 3) == [512, 512, 512]

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.uint8)
    encoded = encode_matrix(image)
    assert isinstance(encoded, list)
    assert encoded[:1] == [int(image.reshape(-1)[0])]
    restored = decode_matrix(encoded)
    assert np.array_equal(restored, image)
    # A query can return the matrix already parsed, or as the text AnyLog prints.
    assert np.array_equal(decode_matrix(json.dumps(encoded)), image)
    assert np.array_equal(decode_matrix(str(encoded)), image)

    upscaled = upscale_nearest(image, DISPLAY_SCALE)
    assert upscaled.shape == (IMAGE_SIZE * DISPLAY_SCALE, IMAGE_SIZE * DISPLAY_SCALE, IMAGE_CHANNELS)
    averaged = box_downsample(upscaled, IMAGE_SIZE)
    assert np.allclose(averaged, image.astype(np.float32) / 255.0)
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(upscaled).save(buffer, format="PNG")
    batch = image_to_model_batch(buffer.getvalue())
    assert batch.shape == (1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert np.allclose(batch[0].transpose(1, 2, 0), image.astype(np.float32) / 255.0)

    labels = np.array([7, 0, 1, 2, 4, 6, 7, 0])
    chosen = select_eval_indices(labels)
    assert chosen[0] == 1 and chosen[7] == 0 and len(chosen) == len(EVAL_LABELS)
    print("medmnist_common self-check ok")


if __name__ == "__main__":
    _self_check()
