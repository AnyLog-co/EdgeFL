"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

"""Shared Shakespeare vocabulary, time windows, and corpus parsing.

The publisher and the data handler both use these helpers so a training
round queries the same timestamp window the publisher filled.
"""

import re
import zlib
from datetime import datetime, timedelta, timezone

# Tiny Shakespeare character set. The GUI encodes prompts with this same string.
CHARSET = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
STOI = {ch: i for i, ch in enumerate(CHARSET)}
ITOS = {i: ch for ch, i in STOI.items()}

SAMPLE_PROMPT = "To be, or not to be, that is the question"
DEFAULT_SEQ_LEN = 40
DEFAULT_ROUND_SECONDS = 60

_SMALL_WORDS = {"of", "the", "and", "de", "du", "d"}
_NAME_PUNCT = set(",;?!&$")


def parse_epoch(value):
    """Parse an epoch timestamp as UTC."""
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
    """Fixed-width UTC timestamp stored on each row and used in SQL."""
    moment = parse_epoch(moment).replace(tzinfo=None)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")


def round_bounds(round_number, epoch_start, duration_seconds):
    """
    Half-open window [start, end) for a 1-based training round.

    Round 1 starts at epoch_start. Each later round begins where the
    previous window ends, so inserted rows stay in one continuous timeline.
    """
    round_number = int(round_number)
    if round_number < 1:
        raise ValueError("round_number starts at 1")
    duration_seconds = float(duration_seconds)
    if duration_seconds <= 0:
        raise ValueError("round duration must be positive")
    epoch_start = parse_epoch(epoch_start)
    start = epoch_start + timedelta(seconds=(round_number - 1) * duration_seconds)
    end = start + timedelta(seconds=duration_seconds)
    return start, end


def max_round_query(db_name, table):
    return (
        f"sql {db_name} format=json and stat=false "
        f"SELECT max(round_number) FROM {table}"
    )


def select_samples_query(db_name, table, round_number, history_rounds=1):
    round_number = int(round_number)
    history_rounds = max(1, int(history_rounds))
    first_round = max(1, round_number - history_rounds + 1)
    if first_round == round_number:
        where = f"round_number = {round_number}"
    else:
        where = f"round_number >= {first_round} AND round_number <= {round_number}"
    return (
        f"sql {db_name} format=json and stat=false "
        f"SELECT timestamp, sequence, label, role, round_number FROM {table} "
        f"WHERE {where}"
    )


def encode_text(text):
    ids = []
    for ch in text:
        if ch not in STOI:
            raise ValueError(f"Character {ch!r} is outside the Shakespeare vocabulary")
        ids.append(STOI[ch])
    return ids


def decode_ids(ids):
    chars = []
    for raw in ids:
        index = int(raw)
        if index not in ITOS:
            raise ValueError(f"Character id {index} is outside the Shakespeare vocabulary")
        chars.append(ITOS[index])
    return "".join(chars)


def pad_ids(ids, seq_len, pad_char=" "):
    seq_len = int(seq_len)
    values = [int(i) for i in ids]
    if len(values) > seq_len:
        return values[-seq_len:]
    if len(values) < seq_len:
        pad = STOI[pad_char]
        values = [pad] * (seq_len - len(values)) + values
    return values


def filter_text(text):
    return "".join(ch for ch in text if ch in STOI)


def is_speaker_line(line):
    """A speaker cue is a short capitalized name ending in a colon."""
    name = line.strip()
    if not name.endswith(":"):
        return False
    name = name[:-1].strip()
    if not name or len(name) > 40 or any(ch in name for ch in _NAME_PUNCT):
        return False
    words = name.replace(".", "").split()
    if not words:
        return False
    for word in words:
        core = word.strip("'")
        if not core or core.lower() in _SMALL_WORDS:
            continue
        if not core[0].isupper():
            return False
    return True


def parse_role_texts(raw_text):
    """
    Group dialogue by speaking role.

    Speaker cues in the Tiny Shakespeare file are preceded by a blank line.
    Lines such as "Come:" inside a speech are kept as dialogue.
    """
    speeches = {}
    current = None
    buffer = []

    def flush():
        if not current or not buffer:
            buffer.clear()
            return
        text = filter_text("\n".join(buffer).strip())
        if text:
            speeches.setdefault(current, []).append(text)
        buffer.clear()

    lines = raw_text.splitlines()
    for index, line in enumerate(lines):
        previous = lines[index - 1] if index else ""
        if is_speaker_line(line) and (index == 0 or not previous.strip()):
            flush()
            current = line.strip()[:-1].strip()
            continue
        if current is not None and line.strip():
            buffer.append(line.strip())
    flush()
    return {role: "\n".join(parts) for role, parts in speeches.items()}


def portion_counts(total, parts):
    """Split `total` rows across `parts` nodes. Counts differ by at most one."""
    total = int(total)
    parts = int(parts)
    if parts < 1:
        raise ValueError("at least one AnyLog node is required")
    if total < parts:
        raise ValueError(
            f"Need at least one row for each of {parts} nodes, got {total}"
        )
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def assign_node(role, node_count):
    if node_count < 1:
        raise ValueError("at least one AnyLog node is required")
    return zlib.crc32(role.encode("utf-8")) % node_count


class CharStream:
    """Walk a role's text, yielding (sequence, next_char) and wrapping at the end."""

    def __init__(self, text, seq_len, stride):
        self.text = text
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.max_start = len(text) - self.seq_len - 1
        if self.max_start < 0:
            raise ValueError("text is shorter than one labeled window")
        self.pos = 0
        self.wraps = 0

    def snapshot(self):
        return (self.pos, self.wraps)

    def restore(self, state):
        self.pos, self.wraps = state

    def next_sample(self):
        if self.pos > self.max_start:
            self.pos = 0
            self.wraps += 1
        start = self.pos
        sequence = self.text[start:start + self.seq_len]
        label = self.text[start + self.seq_len]
        self.pos += self.stride
        return sequence, label


class RoleCorpus:
    def __init__(self, role, text, seq_len, stride, test_fraction):
        self.role = role
        seq_len = int(seq_len)
        minimum = seq_len + 1
        if len(text) < minimum * 2:
            raise ValueError(f"{role} is too short to split into train and test")
        test_fraction = float(test_fraction)
        split = int(len(text) * (1.0 - test_fraction))
        split = min(max(split, minimum), len(text) - minimum)
        self.train = CharStream(text[:split], seq_len, stride)
        self.test = CharStream(text[split:], seq_len, stride)


class NodeFeed:
    def __init__(self, roles):
        if not roles:
            raise ValueError("node has no speaking roles")
        self.roles = roles
        self._train_i = 0
        self._test_i = 0

    def snapshot(self):
        return (
            self._train_i,
            self._test_i,
            [(role.train.snapshot(), role.test.snapshot()) for role in self.roles],
        )

    def restore(self, state):
        self._train_i, self._test_i, streams = state
        for role, (train_state, test_state) in zip(self.roles, streams):
            role.train.restore(train_state)
            role.test.restore(test_state)

    def next_train(self):
        role = self.roles[self._train_i % len(self.roles)]
        self._train_i += 1
        sequence, label = role.train.next_sample()
        return role.role, sequence, label

    def next_test(self):
        role = self.roles[self._test_i % len(self.roles)]
        self._test_i += 1
        sequence, label = role.test.next_sample()
        return role.role, sequence, label


def build_node_feeds(raw_text, node_count, seq_len, stride, test_fraction):
    """Partition speaking roles across AnyLog nodes and build continuous cursors."""
    role_texts = parse_role_texts(raw_text)
    grouped = [[] for _ in range(node_count)]
    skipped = []
    for role, text in sorted(role_texts.items()):
        try:
            corpus = RoleCorpus(role, text, seq_len, stride, test_fraction)
        except ValueError:
            skipped.append(role)
            continue
        grouped[assign_node(role, node_count)].append(corpus)

    empty = [index for index, roles in enumerate(grouped) if not roles]
    if empty:
        raise ValueError(
            "Some nodes received no speaking roles. Use fewer nodes or a larger corpus. "
            f"Empty node indexes: {empty}"
        )
    feeds = [NodeFeed(roles) for roles in grouped]
    return feeds, {"roles_kept": sum(len(roles) for roles in grouped), "roles_skipped": len(skipped)}


def scheduled_times(start, end, count):
    """Spread `count` timestamps across the open side of the round window."""
    count = int(count)
    if count <= 0:
        return []
    start = parse_epoch(start)
    end = parse_epoch(end)
    span = (end - start).total_seconds() * 0.98
    if count == 1:
        return [start + timedelta(seconds=span / 2.0)]
    step = span / (count - 1)
    return [start + timedelta(seconds=step * i) for i in range(count)]


def _self_check():
    epoch = parse_epoch("2026-10-01T17:00:00Z")
    start, end = round_bounds(1, epoch, 60)
    next_start, next_end = round_bounds(2, epoch, 60)
    assert end == next_start
    assert (next_end - next_start).total_seconds() == 60
    assert format_ts(start) == "2026-10-01 17:00:00.000000"
    ids = encode_text(SAMPLE_PROMPT)
    assert decode_ids(ids) == SAMPLE_PROMPT
    padded = pad_ids(ids[:3], 8)
    assert len(padded) == 8
    query = select_samples_query("shakespeare_fl", "shakespeare_train", 1)
    assert "WHERE round_number = 1" in query
    span = select_samples_query("shakespeare_fl", "shakespeare_train", 12, 3)
    assert "round_number >= 10 AND round_number <= 12" in span
    assert "max(round_number)" in max_round_query("shakespeare_fl", "shakespeare_test")
    assert portion_counts(64, 3) == [22, 21, 21]
    assert sum(portion_counts(16, 3)) == 16
    raw = "\n".join([
        "HAMLET:",
        "To be, or not to be, that is the question",
        "",
        "HORATIO:",
        "I knew him, Horatio",
    ])
    roles = parse_role_texts(raw)
    assert "HAMLET" in roles and "HORATIO" in roles
    times = scheduled_times(start, end, 4)
    assert times[0] >= start and times[-1] < end
    print("shakespeare_common self-check ok")


if __name__ == "__main__":
    _self_check()
