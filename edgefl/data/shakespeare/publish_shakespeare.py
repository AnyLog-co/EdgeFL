"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""
Stream labeled Shakespeare lines to a set of AnyLog nodes.

Each round's labeled rows are split evenly across the configured AnyLog
nodes, so every operator receives its own portion. The publisher walks
the play in order, labels the next character, and inserts that node's
rows across the round's time window. It keeps going until stopped,
wrapping to the start of the text when a role runs out.

Each inserted row includes round_number. Training nodes select
WHERE round_number = N. DATA_EPOCH_START and ROUND_DURATION_SECONDS
still pace the inserts across each round.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_components.data_handlers.shakespeare_common import (  # noqa: E402
    DEFAULT_ROUND_SECONDS,
    DEFAULT_SEQ_LEN,
    build_node_feeds,
    format_ts,
    parse_epoch,
    portion_counts,
    round_bounds,
    scheduled_times,
    select_samples_query,
)

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
STATE_PATH = Path(__file__).resolve().parent / "publish_state.json"
DEFAULT_CORPUS = Path(__file__).resolve().parent / "input.txt"


def load_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config file must be a JSON object")
    return payload


def normalize_nodes(value):
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    nodes = []
    for part in parts:
        node = str(part).strip()
        if not node:
            continue
        for prefix in ("http://", "https://"):
            if node.startswith(prefix):
                node = node[len(prefix):]
        nodes.append(node.rstrip("/"))
    return nodes


def ensure_corpus(path, url):
    path = Path(path)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Shakespeare corpus to {path}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def create_header(db_name, table_name):
    return {
        "type": "json",
        "dbms": db_name,
        "table": table_name,
        "mode": "streaming",
        "Content-Type": "text/plain",
    }


def sql_text(value):
    """AnyLog wraps string values in single quotes and does not escape them."""
    return str(value).replace("'", "''")


def prepare_node(conn, db_name, tables):
    """Refresh AnyLog table metadata the same way the other dataset loaders do."""
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
    # A 60s flush is the same length as a round, so the query at the end of
    # an odd round runs while that round is still sitting in the buffer.
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


def put_row(conn, db_name, table_name, row):
    response = requests.put(
        url=f"http://{conn}",
        data=json.dumps(row),
        headers=create_header(db_name, table_name),
        timeout=30,
    )
    response.raise_for_status()


def sleep_until(moment):
    while True:
        remaining = (moment - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def write_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote publisher state to {STATE_PATH}")


def build_events(feed, nodes, start, end, samples_per_round, test_samples):
    """Deal this round's rows round-robin so every node gets its own timestamps."""
    events = []
    node_count = len(nodes)
    draws = (
        ("train", samples_per_round, feed.next_train),
        ("test", test_samples, feed.next_test),
    )
    for split, total, take_sample in draws:
        for offset, moment in enumerate(scheduled_times(start, end, total)):
            index = offset % node_count
            role, sequence, label = take_sample()
            events.append((moment, nodes[index], split, role, sequence, label, index))
    events.sort(key=lambda item: (item[0], item[6], item[2]))
    return events


def stream_round(round_number, feed, nodes, args, tables):
    start, end = round_bounds(round_number, args.epoch_start, args.round_duration)
    events = build_events(
        feed,
        nodes,
        start,
        end,
        args.samples_per_round,
        args.test_samples_per_round,
    )
    failures = 0
    sent = {conn: {"train": 0, "test": 0} for conn in nodes}
    train_counts = portion_counts(args.samples_per_round, len(nodes))
    test_counts = portion_counts(args.test_samples_per_round, len(nodes))
    print(
        f"Round {round_number}: streaming {len(events)} labeled rows "
        f"across [{format_ts(start)}, {format_ts(end)})"
    )
    for conn, train_count, test_count in zip(nodes, train_counts, test_counts):
        print(f"  {conn}: train {train_count}, test {test_count}")
    for moment, conn, split, role, sequence, label, _index in events:
        sleep_until(moment)
        row = {
            "timestamp": format_ts(moment),
            "role": sql_text(role),
            "sequence": sql_text(sequence),
            "label": sql_text(label),
            "round_number": round_number,
        }
        table = tables[split]
        try:
            put_row(conn, args.db_name, table, row)
            sent[conn][split] += 1
        except Exception as error:
            failures += 1
            print(f"Failed to insert {split} row for {role} on {conn}: {error}")
    sleep_until(end)
    for conn in nodes:
        print(
            f"  published to {conn}: train={sent[conn]['train']} test={sent[conn]['test']}"
        )
    print(f"Round {round_number} complete. failures={failures}")
    return failures


def print_plan(feed, nodes, args):
    saved = feed.snapshot()
    try:
        start, end = round_bounds(1, args.epoch_start, args.round_duration)
        train_counts = portion_counts(args.samples_per_round, len(nodes))
        test_counts = portion_counts(args.test_samples_per_round, len(nodes))
        print(f"Nodes: {', '.join(nodes)}")
        print(
            f"Roles kept: {args.corpus_stats['roles_kept']} "
            f"(skipped {args.corpus_stats['roles_skipped']} short roles)"
        )
        print(
            f"Each round splits {args.samples_per_round} train rows and "
            f"{args.test_samples_per_round} test rows across {len(nodes)} nodes"
        )
        for conn, train_count, test_count in zip(nodes, train_counts, test_counts):
            print(f"  {conn}: train {train_count}, test {test_count}")
        print("Round 1 train query:")
        print(f"  {select_samples_query(args.db_name, args.train_table, 1)}")
        preview = build_events(
            feed,
            nodes,
            start,
            end,
            args.samples_per_round,
            args.test_samples_per_round,
        )
        print("First scheduled inserts:")
        for moment, conn, split, role, sequence, label, _index in preview[:6]:
            print(f"  {format_ts(moment)} {split} {conn} {role!r} -> {label!r} context={sequence[-24:]!r}")
    finally:
        feed.restore(saved)


def parse_args(argv):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=None)
    known, _remaining = preliminary.parse_known_args(argv)
    config = load_config(known.config)

    parse = argparse.ArgumentParser(description="Publish labeled Shakespeare rows to AnyLog nodes")
    parse.add_argument("--config", default=None, help="JSON file with publisher settings")
    parse.add_argument("--nodes", default=config.get("nodes"), help="Comma-separated AnyLog REST host:port list")
    parse.add_argument("--db-name", default=config.get("db_name", "shakespeare_fl"))
    parse.add_argument("--train-table", default=config.get("train_table", "shakespeare_train"))
    parse.add_argument("--test-table", default=config.get("test_table", "shakespeare_test"))
    parse.add_argument("--epoch-start", default=config.get("epoch_start"), help="UTC timestamp for round 1. Defaults to now.")
    parse.add_argument("--round-duration", type=float, default=config.get("round_duration_seconds", DEFAULT_ROUND_SECONDS))
    parse.add_argument("--samples-per-round", type=int, default=config.get("samples_per_round", 64))
    parse.add_argument("--test-samples-per-round", type=int, default=config.get("test_samples_per_round"))
    parse.add_argument("--seq-len", type=int, default=config.get("seq_len", DEFAULT_SEQ_LEN))
    parse.add_argument("--stride", type=int, default=config.get("stride", 20))
    parse.add_argument("--test-fraction", type=float, default=config.get("test_fraction", 0.2))
    parse.add_argument("--num-rounds", type=int, default=config.get("num_rounds", 0), help="0 streams until interrupted")
    parse.add_argument("--corpus", default=config.get("corpus", str(DEFAULT_CORPUS)))
    parse.add_argument("--corpus-url", default=config.get("corpus_url", CORPUS_URL))
    parse.add_argument("--dry-run", action="store_true", help="Parse, partition, and print the round plan")
    args = parse.parse_args(argv)

    args.nodes = normalize_nodes(args.nodes)
    if not args.nodes:
        parse.error("specify AnyLog nodes with --nodes or the config file")
    if args.samples_per_round < 1:
        parse.error("--samples-per-round must be at least 1")
    if args.test_samples_per_round is None:
        args.test_samples_per_round = max(1, int(args.samples_per_round * args.test_fraction))
    if args.epoch_start:
        args.epoch_start = parse_epoch(args.epoch_start)
    else:
        args.epoch_start = datetime.now(timezone.utc)
    return args


def main(argv=None):
    args = parse_args(argv)
    raw_text = ensure_corpus(args.corpus, args.corpus_url)
    feeds, stats = build_node_feeds(
        raw_text,
        1,
        args.seq_len,
        args.stride,
        args.test_fraction,
    )
    feed = feeds[0]
    args.corpus_stats = stats
    print(f"DATA_EPOCH_START={args.epoch_start.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"ROUND_DURATION_SECONDS={args.round_duration}")
    print(f"SEQ_LEN={args.seq_len}")
    print_plan(feed, args.nodes, args)

    if args.dry_run:
        return 0

    state = {
        "DATA_EPOCH_START": args.epoch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ROUND_DURATION_SECONDS": args.round_duration,
        "SEQ_LEN": args.seq_len,
        "LOGICAL_DATABASE": args.db_name,
        "TRAIN_TABLE": args.train_table,
        "TEST_TABLE": args.test_table,
        "nodes": args.nodes,
    }
    write_state(state)

    tables = {"train": args.train_table, "test": args.test_table}
    for conn in args.nodes:
        prepare_node(conn, args.db_name, (args.train_table, args.test_table))

    round_number = 1
    try:
        while args.num_rounds <= 0 or round_number <= args.num_rounds:
            stream_round(round_number, feed, args.nodes, args, tables)
            round_number += 1
    except KeyboardInterrupt:
        print(f"Publisher stopped during round {round_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
