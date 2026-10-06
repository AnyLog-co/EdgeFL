"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""
Publish BloodMNIST image info to AnyLog / EdgeLake operators.

This follows the chest X-ray publisher: each row is scalar metadata
(filename, width, height, label, class_name, round_number). The PNG itself
is written under published_images/ and is not sent to AnyLog. Training nodes
query the filename and open that file.

    python3 publish_medmnist_info.py 127.0.0.1:32149,127.0.0.1:32249,127.0.0.1:32349

publish_medmnist.py is the earlier publisher that inserts the pixel matrix.
It is kept as a record. Use this file for the demo.
"""

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_components.data_handlers.medmnist_common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_SAMPLES_PER_ROUND,
    DEFAULT_TEST_SAMPLES_PER_ROUND,
    format_ts,
    portion_counts,
    select_image_info_query,
)
from publish_medmnist import (  # noqa: E402
    build_events,
    build_shards,
    hold_out_eval_images,
    normalize_nodes,
    prepare_node,
    put_row,
)
from dataset_io import load_train_and_test  # noqa: E402

# Same directory IMAGE_ROOT_DIR points at in the node env files.
IMAGE_DIR = Path(__file__).resolve().parent / "published_images"
INSERT_WORKERS = 8
_WRITE_LOCK = threading.Lock()


def slide_filename(split, image, label):
    """Stable name for one slide. A later round that repeats the slide reuses the file."""
    array = np.ascontiguousarray(image, dtype=np.uint8)
    digest = hashlib.sha1(bytes([int(label)]) + array.tobytes()).hexdigest()[:16]
    return f"{split}_{int(label)}_{digest}.png"


def ensure_slide(directory, filename, image):
    """Write the PNG once. AnyLog only receives the filename."""
    path = directory / filename
    if path.exists():
        return
    array = np.ascontiguousarray(image, dtype=np.uint8)
    with _WRITE_LOCK:
        if path.exists():
            return
        Image.fromarray(array).save(path)


def image_info_row(split, image, label, round_number):
    """Scalar columns, matching the chest X-ray rows (filename, size, class, round)."""
    filename = slide_filename(split, image, label)
    ensure_slide(IMAGE_DIR, filename, image)
    height, width = int(image.shape[0]), int(image.shape[1])
    return {
        "timestamp": format_ts(datetime.now(timezone.utc)),
        "filename": filename,
        "width": width,
        "height": height,
        "label": int(label),
        "class_name": CLASS_NAMES[int(label)],
        "round_number": int(round_number),
    }


def stream_round(round_number, train_shards, test_shards, nodes, args, tables):
    """Insert this round's filenames immediately. Training selects on round_number."""
    train_counts = portion_counts(args.samples_per_round, len(nodes))
    test_counts = portion_counts(args.test_samples_per_round, len(nodes))
    events = build_events(train_shards, test_shards, nodes, train_counts, test_counts)
    failures = 0
    sent = {conn: {"train": 0, "test": 0} for conn in nodes}
    print(f"Round {round_number}: inserting {len(events)} image-info rows")
    for conn, train_count, test_count in zip(nodes, train_counts, test_counts):
        print(f"  {conn}: train {train_count}, test {test_count}")

    rows = []
    for conn, split, image, label in events:
        rows.append((conn, split, image_info_row(split, image, label, round_number)))

    def send(item):
        conn, split, row = item
        put_row(conn, args.db_name, tables[split], row)
        return conn, split

    workers = min(INSERT_WORKERS, max(1, len(rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(send, item) for item in rows]
        for future in as_completed(futures):
            try:
                conn, split = future.result()
                sent[conn][split] += 1
            except Exception as error:
                failures += 1
                print(f"Failed to insert row: {error}")
    for conn in nodes:
        print(f"  published to {conn}: train={sent[conn]['train']} test={sent[conn]['test']}")
    print(f"Round {round_number} complete. failures={failures}")
    return failures


def print_plan(nodes, args, eval_indices):
    train_counts = portion_counts(args.samples_per_round, len(nodes))
    test_counts = portion_counts(args.test_samples_per_round, len(nodes))
    print(f"Operators: {', '.join(nodes)}")
    print(f"Image files: {IMAGE_DIR}")
    print(
        f"Each round inserts {args.samples_per_round} train rows and "
        f"{args.test_samples_per_round} test rows"
    )
    for conn, train_count, test_count in zip(nodes, train_counts, test_counts):
        print(f"  {conn}: train {train_count}, test {test_count}")
    print("Held-out eval slides (not inserted):")
    for label in sorted(eval_indices):
        print(f"  test[{eval_indices[label]}] label {label} ({CLASS_NAMES[label]})")
    print("Round 1 train query:")
    print(f"  {select_image_info_query(args.db_name, args.train_table, 1)}")
    example = {
        "filename": "train_6_0123456789abcdef.png",
        "width": 28,
        "height": 28,
        "label": 6,
        "class_name": CLASS_NAMES[6],
        "round_number": 1,
    }
    print(f"Example row: {json.dumps(example)}")


def parse_args(argv):
    parse = argparse.ArgumentParser(
        description="Publish BloodMNIST filenames and labels to AnyLog operators. "
                    "The PNG stays on disk. Pass REST addresses as host:port,host:port."
    )
    parse.add_argument(
        "nodes",
        help="Comma-separated AnyLog/EdgeLake REST addresses, for example "
             "127.0.0.1:32149,127.0.0.1:32249,127.0.0.1:32349",
    )
    parse.add_argument("--db-name", default="mydb")
    parse.add_argument("--train-table", default="medmnist_train")
    parse.add_argument("--test-table", default="medmnist_test")
    parse.add_argument("--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND)
    parse.add_argument("--test-samples-per-round", type=int, default=DEFAULT_TEST_SAMPLES_PER_ROUND)
    parse.add_argument(
        "--num-rounds",
        type=int,
        default=0,
        help="0 keeps streaming until the process is interrupted",
    )
    parse.add_argument("--seed", type=int, default=7)
    parse.add_argument("--dry-run", action="store_true", help="Print the round plan and exit")
    args = parse.parse_args(argv)
    try:
        args.nodes = normalize_nodes(args.nodes)
    except ValueError as error:
        parse.error(str(error))
    if not args.nodes:
        parse.error("pass at least one host:port")
    if args.samples_per_round < len(args.nodes):
        parse.error("--samples-per-round must cover every operator")
    if args.test_samples_per_round < len(args.nodes):
        parse.error("--test-samples-per-round must cover every operator")
    return args


def main(argv=None):
    args = parse_args(argv)
    train_images, train_labels, test_images, test_labels = load_train_and_test()
    test_images, test_labels, eval_indices = hold_out_eval_images(test_images, test_labels)
    train_shards = build_shards(train_images, train_labels, len(args.nodes), args.seed)
    test_shards = build_shards(test_images, test_labels, len(args.nodes), args.seed + 1)

    print(f"TRAIN_IMAGES={len(train_labels)} TEST_IMAGES={len(test_labels)}")
    print_plan(args.nodes, args, eval_indices)
    if args.dry_run:
        return 0

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = Path(__file__).resolve().parent / "publish_state.json"
    state = {
        "publisher": "publish_medmnist_info.py",
        "IMAGE_DIR": str(IMAGE_DIR),
        "LOGICAL_DATABASE": args.db_name,
        "TRAIN_TABLE": args.train_table,
        "TEST_TABLE": args.test_table,
        "SAMPLES_PER_ROUND": args.samples_per_round,
        "TEST_SAMPLES_PER_ROUND": args.test_samples_per_round,
        "nodes": args.nodes,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote publisher state to {state_path}")

    tables = {"train": args.train_table, "test": args.test_table}
    for conn in args.nodes:
        prepare_node(conn, args.db_name, (args.train_table, args.test_table))

    round_number = 1
    try:
        while args.num_rounds <= 0 or round_number <= args.num_rounds:
            stream_round(round_number, train_shards, test_shards, args.nodes, args, tables)
            round_number += 1
    except KeyboardInterrupt:
        print(f"Publisher stopped during round {round_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
