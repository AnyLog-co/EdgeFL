"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""
Write six BloodMNIST test slides into eval_images/.

The file name is the correct class (`6_neutrophil.png`). Pixels are
nearest-neighbor scaled so the picture is large enough to inspect, and the
same block average the GUI uses restores the original 28x28 image.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_io import load_train_and_test  # noqa: E402
from platform_components.data_handlers.medmnist_common import (  # noqa: E402
    CLASS_NAMES,
    DISPLAY_SCALE,
    EVAL_LABELS,
    IMAGE_SIZE,
    box_downsample,
    file_stem,
    select_eval_indices,
    upscale_nearest,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "eval_images"


def export_images(output_dir=OUTPUT_DIR):
    _train_images, _train_labels, test_images, test_labels = load_train_and_test()
    chosen = select_eval_indices(test_labels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "bloodmnist",
        "source_split": "test",
        "image_size": IMAGE_SIZE,
        "display_scale": DISPLAY_SCALE,
        "display_size": IMAGE_SIZE * DISPLAY_SCALE,
        "note": "The file name is the correct prediction. Upload the PNG in the GUI.",
        "images": [],
    }

    for label in EVAL_LABELS:
        index = chosen[label]
        source = test_images[index]
        upscaled = upscale_nearest(source, DISPLAY_SCALE)
        restored = box_downsample(upscaled, IMAGE_SIZE)
        if not np.allclose(restored, source.astype(np.float32) / 255.0):
            raise RuntimeError(f"Eval image for {CLASS_NAMES[label]} did not round-trip to 28x28")
        filename = f"{file_stem(label)}.png"
        Image.fromarray(upscaled, mode="RGB").save(output_dir / filename)
        manifest["images"].append({
            "file": filename,
            "label": int(label),
            "class_name": CLASS_NAMES[label],
            "file_stem": file_stem(label),
            "test_index": int(index),
        })
        print(f"Wrote {filename} (test index {index})")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    export_images()
