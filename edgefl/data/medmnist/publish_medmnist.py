"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""
Stream labeled BloodMNIST slides to AnyLog / EdgeLake operators.

The only required input is the comma-separated REST address of each operator:

    python3 publish_medmnist.py 127.0.0.1:32149,127.0.0.1:32249,127.0.0.1:32349

Each operator receives its own class-balanced shard. The process keeps
inserting rows until it is stopped. When a shard runs out, that operator
starts again at the beginning of its shard, so later rounds still have data.

Each inserted row stores the 28x28x3 pixel matrix as a numeric array, the
class label, and round_number. It does not store an image file or a blob.
Training nodes select WHERE round_number = N and read `matrix`. A round is
inserted immediately instead of being spread across a clock window.
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_io import load_train_and_test  # noqa: E402
from platform_components.data_handlers.medmnist_common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_SAMPLES_PER_ROUND,
    DEFAULT_TEST_SAMPLES_PER_ROUND,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    encode_matrix,
    format_ts,
    portion_counts,
    select_eval_indices,
    select_images_query,
)

# Parallel REST puts. Selection is by round_number, so order inside a round
# does not matter and the operators can be filled at the same time.
INSERT_WORKERS = 8
_THREAD_STATE = threading.local()

STATE_PATH = Path(__file__).resolve().parent / "publish_state.json"


def normalize_nodes(value):
    """Accept 'host:port, host:port' with or without an http:// prefix."""
    if value is None:
        return []
    nodes = []
    for part in str(value).split(","):
        node = part.strip()
        if not node:
            continue
        for prefix in ("http://", "https://"):
            if node.startswith(prefix):
                node = node[len(prefix):]
        node = node.rstrip("/")
        if ":" not in node:
            raise ValueError(f"Expected host:port, got {node!r}")
        nodes.append(node)
    return nodes


def create_header(db_name, table_name):
    return {
        "type": "json",
        "dbms": db_name,
        "table": table_name,
        "mode": "streaming",
        "Content-Type": "text/plain",
    }


def prepare_node(conn, db_name, tables):
    """
    Ask AnyLog to flush these tables quickly.

    Training selects on round_number as soon as the round is inserted, so a
    long buffer would hide rows the node is already querying.
    """
    for command in ("drop", "create"):
        try:
            response = requests.post(
                f"http://{conn}",
                headers={
                    "command": f"{command} table tsd_info where dbms=almgm",
                    "User-Agent": "AnyLog/1.23",
                },
                timeout=30,
            )
            response.raise_for_status()
        except Exception as error:
            print(f"Warning: could not {command} tsd_info on {conn} ({error})")
    for table in tables:
        command = (
            f"set buffer threshold where dbms = {db_name} and table = {table} "
            f"and time = 1 second and write_immediate = true"
        )
        try:
            response = requests.post(
                f"http://{conn}",
                headers={"command": command, "User-Agent": "AnyLog/1.23"},
                timeout=30,
            )
            response.raise_for_status()
        except Exception as error:
            print(f"Warning: could not set a 1s flush for {table} on {conn} ({error})")


def _session():
    """One connection pool per worker thread. requests.Session is not shared."""
    session = getattr(_THREAD_STATE, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_STATE.session = session
    return session


def put_row(conn, db_name, table_name, row):
    response = _session().put(
        url=f"http://{conn}",
        data=json.dumps(row),
        headers=create_header(db_name, table_name),
        timeout=30,
    )
    response.raise_for_status()


def write_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote publisher state to {STATE_PATH}")


class ClassCycle:
    """Walk one class's images and wrap to the start when the shard runs out."""

    def __init__(self, images):
        self.images = images
        self.cursor = 0

    def take(self, count):
        picked = []
        length = len(self.images)
        if length == 0 or count <= 0:
            return picked
        for _ in range(count):
            picked.append(self.images[self.cursor % length])
            self.cursor += 1
        return picked


class NodeShard:
    """
    One operator's images, split by class.

    take() deals the requested count across classes so a round is not a long
    run of a single cell type. The cursor survives across rounds.
    """

    def __init__(self, images, labels):
        self.cycles = {}
        # Rotates which classes are used when a round asks for fewer images than classes.
        self.rotation = 0
        flat_labels = np.asarray(labels).reshape(-1)
        for class_id in range(len(CLASS_NAMES)):
            self.cycles[class_id] = ClassCycle(images[flat_labels == class_id])

    def take(self, count):
        active = [class_id for class_id, cycle in self.cycles.items() if len(cycle.images)]
        if not active:
            raise RuntimeError("Operator shard has no images")
        if count < len(active):
            start = self.rotation % len(active)
            self.rotation += count
            return [
                (self.cycles[active[(start + offset) % len(active)]].take(1)[0],
                 active[(start + offset) % len(active)])
                for offset in range(count)
            ]
        counts = portion_counts(count, len(active))
        columns = []
        for class_id, class_count in zip(active, counts):
            columns.append([(image, class_id) for image in self.cycles[class_id].take(class_count)])
        # Interleave classes so inserts for one cell type are not bunched together.
        samples = []
        depth = max(len(column) for column in columns)
        for offset in range(depth):
            for column in columns:
                if offset < len(column):
                    samples.append(column[offset])
        return samples


def build_shards(images, labels, node_count, seed):
    """
    Give every operator a class-balanced slice.

    Within each class, rows are shuffled once and then dealt round-robin.
    Operator 0 does not see operator 1's slides. That is the federated split.
    """
    rng = np.random.default_rng(seed)
    buckets = [
        [[] for _ in range(len(CLASS_NAMES))]
        for _ in range(node_count)
    ]
    flat_labels = np.asarray(labels).reshape(-1)
    for class_id in range(len(CLASS_NAMES)):
        indexes = np.flatnonzero(flat_labels == class_id)
        rng.shuffle(indexes)
        for offset, index in enumerate(indexes.tolist()):
            buckets[offset % node_count][class_id].append(images[index])

    shards = []
    for node_buckets in buckets:
        node_images = []
        node_labels = []
        for class_id, class_images in enumerate(node_buckets):
            node_images.extend(class_images)
            node_labels.extend([class_id] * len(class_images))
        if node_images:
            stacked = np.stack(node_images)
        else:
            stacked = np.empty((0, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.uint8)
        shards.append(NodeShard(stacked, np.asarray(node_labels, dtype=np.int64)))
    return shards


def hold_out_eval_images(test_images, test_labels):
    """Drop the six slides exported under eval_images/ so they stay unseen."""
    chosen = select_eval_indices(test_labels)
    keep = np.ones(len(test_labels), dtype=bool)
    keep[np.fromiter(chosen.values(), dtype=np.int64)] = False
    return test_images[keep], test_labels[keep], chosen


def build_events(train_shards, test_shards, nodes, train_counts, test_counts):
    """Pair each sample with the operator that owns it. No clock is involved."""
    planned = []
    for index, conn in enumerate(nodes):
        planned.append((train_counts[index], train_shards[index], "train", conn))
        planned.append((test_counts[index], test_shards[index], "test", conn))
    events = []
    for count, shard, split, conn in planned:
        samples = shard.take(count)
        if len(samples) != count:
            raise RuntimeError(f"Expected {count} {split} images for {conn}, got {len(samples)}")
        for image, label in samples:
            events.append((conn, split, image, label))
    return events


def stream_round(round_number, train_shards, test_shards, nodes, args, tables):
    """Insert this round's matrices immediately. Training selects on round_number."""
    train_counts = portion_counts(args.samples_per_round, len(nodes))
    test_counts = portion_counts(args.test_samples_per_round, len(nodes))
    events = build_events(train_shards, test_shards, nodes, train_counts, test_counts)
    failures = 0
    sent = {conn: {"train": 0, "test": 0} for conn in nodes}
    print(f"Round {round_number}: inserting {len(events)} labeled matrices")
    for conn, train_count, test_count in zip(nodes, train_counts, test_counts):
        print(f"  {conn}: train {train_count}, test {test_count}")

    def send(event):
        conn, split, image, label = event
        row = {
            "timestamp": format_ts(datetime.now(timezone.utc)),
            # Numeric 28x28x3 matrix, not an image file and not a JSON blob.
            "matrix": encode_matrix(image),
            "label": int(label),
            "round_number": round_number,
        }
        put_row(conn, args.db_name, tables[split], row)
        return conn, split

    workers = min(INSERT_WORKERS, max(1, len(events)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(send, event) for event in events]
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
    print(
        f"Each round inserts {args.samples_per_round} train images and "
        f"{args.test_samples_per_round} test images"
    )
    for conn, train_count, test_count in zip(nodes, train_counts, test_counts):
        print(f"  {conn}: train {train_count}, test {test_count}")
    print("Held-out eval slides (not inserted):")
    for label in sorted(eval_indices):
        print(f"  test[{eval_indices[label]}] label {label} ({CLASS_NAMES[label]})")
    print("Round 1 train query:")
    print(f"  {select_images_query(args.db_name, args.train_table, 1)}")


def parse_args(argv):
    parse = argparse.ArgumentParser(
        description="Publish BloodMNIST images to AnyLog operators. "
                    "Pass their REST addresses as host:port,host:port."
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

    state = {
        "LOGICAL_DATABASE": args.db_name,
        "TRAIN_TABLE": args.train_table,
        "TEST_TABLE": args.test_table,
        "SAMPLES_PER_ROUND": args.samples_per_round,
        "TEST_SAMPLES_PER_ROUND": args.test_samples_per_round,
        "nodes": args.nodes,
    }
    write_state(state)

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
