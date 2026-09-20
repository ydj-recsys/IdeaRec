#!/usr/bin/env python3


from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
import tempfile
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
AMAZON = ("baby", "beauty", "sports", "toys")
DATASETS = AMAZON + ("yelp", "ml-100k")
ARTIFACTS = ("dataset.pkl", "item2id.json", "id2item.json", "user2id.json", "stats.json")


@dataclass(frozen=True, slots=True)
class Interaction:
    user: str
    item: str
    rating: float | None
    timestamp: int
    order: int


def iter_lines(path: Path, counts: Counter) -> Iterator[tuple[int, str]]:
    """Read one line at a time, for either plain text or gzip; never read all text."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        for order, raw in enumerate(stream):
            counts["lines"] += 1
            if counts["lines"] % 250000 == 0:
                print(f"  {path.name}: {counts['lines']:,} lines read", flush=True)
            if not raw.strip():
                counts["blank_lines"] += 1
                continue
            try:
                line = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                counts["invalid_utf8"] += 1
                continue
            yield order, line


def iter_json_records(path: Path, counts: Counter, literal_fallback: bool = False
                      ) -> Iterator[tuple[int, dict]]:
    for order, line in iter_lines(path, counts):
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            if not literal_fallback:
                counts["malformed_lines"] += 1
                continue
            try:
                record = ast.literal_eval(line)  # safe old-Amazon dict compatibility
                counts["literal_eval_lines"] += 1
            except (ValueError, SyntaxError, TypeError, RecursionError, OverflowError):
                counts["malformed_lines"] += 1
                continue
        if not isinstance(record, dict):
            counts["non_dict_lines"] += 1
            continue
        counts["parsed_records"] += 1
        yield order, record


def clean_id(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_rating(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def parse_timestamp(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def parse_yelp_date(value: object) -> int | None:
    """Yelp's timezone-free dates are consistently interpreted as UTC, not local."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        date = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return int(date.timestamp())
    except (ValueError, OverflowError, OSError):
        return None


def make_interaction(user: object, item: object, rating: object, timestamp: int | None,
                     order: int, counts: Counter) -> Interaction | None:
    user_id, item_id = clean_id(user), clean_id(item)
    if user_id is None or item_id is None or timestamp is None:
        counts["invalid_required_fields"] += 1
        counts["invalid_user"] += user_id is None
        counts["invalid_item"] += item_id is None
        counts["invalid_timestamp"] += timestamp is None
        return None
    numeric_rating = parse_rating(rating)
    counts["missing_or_invalid_rating"] += numeric_rating is None
    counts["valid_interactions"] += 1
    return Interaction(user_id, item_id, numeric_rating, timestamp, order)


def load_amazon(path: Path, counts: Counter) -> Iterator[Interaction]:
    for order, record in iter_json_records(path, counts, literal_fallback=True):
        row = make_interaction(record.get("reviewerID"), record.get("asin"),
                               record.get("overall"), parse_timestamp(record.get("unixReviewTime")),
                               order, counts)
        if row is not None:
            yield row


def load_yelp(path: Path, counts: Counter) -> Iterator[Interaction]:
    for order, record in iter_json_records(path, counts):
        row = make_interaction(record.get("user_id"), record.get("business_id"),
                               record.get("stars"), parse_yelp_date(record.get("date")), order, counts)
        if row is not None:
            yield row


def load_movielens(path: Path, counts: Counter) -> Iterator[Interaction]:
    for order, line in iter_lines(path, counts):
        # Preserve an empty rating field in tab-separated input as rating=None.
        fields = line.rstrip("\r\n").split("\t") if "\t" in line else line.split()
        if len(fields) != 4:
            counts["malformed_lines"] += 1
            continue
        counts["parsed_records"] += 1
        row = make_interaction(fields[0], fields[1], fields[2],
                               parse_timestamp(fields[3]), order, counts)
        if row is not None:
            yield row


def load_raw_data(dataset: str, path: Path, counts: Counter) -> Iterator[Interaction]:
    if dataset in AMAZON:
        return load_amazon(path, counts)
    if dataset == "yelp":
        return load_yelp(path, counts)
    if dataset == "ml-100k":
        return load_movielens(path, counts)
    raise ValueError(f"Unsupported dataset: {dataset}")


def filter_rating(rows: Iterable[Interaction], min_rating: float | None,
                  counts: Counter) -> Iterator[Interaction]:
    for row in rows:
        if min_rating is None or row.rating is None or row.rating >= min_rating:
            counts["after_rating_filter"] += 1
            yield row
        else:
            counts["removed_by_rating"] += 1


def date_argument(value: str) -> str:
    """A calendar date with explicit UTC, whole-day CLI semantics."""
    try:
        date = datetime.strptime(value, "%Y-%m-%d")
        if date.date().isoformat() != value:
            raise ValueError("Use zero-padded ISO dates")
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a valid YYYY-MM-DD date.") from error
    return value


def filter_dates(rows: Iterable[Interaction], start_date: str | None,
                 end_date: str | None, counts: Counter) -> Iterator[Interaction]:
    """Apply an optional inclusive UTC date range before other filters."""
    if start_date is None and end_date is None:
        yield from rows
        return
    start = (int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
             if start_date else None)
    end_exclusive = (int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400
                     if end_date else None)
    for row in rows:
        if ((start is None or row.timestamp >= start)
                and (end_exclusive is None or row.timestamp < end_exclusive)):
            counts["after_date_filter"] += 1
            yield row
        else:
            counts["removed_by_date"] += 1


def deduplicate(rows: Iterable[Interaction], strategy: str, counts: Counter) -> list[Interaction]:
    if strategy not in ("none", "earliest", "latest"):
        raise ValueError(f"Unknown dedup strategy: {strategy}")
    selected: dict[tuple[str, str], Interaction] = {}
    retained: list[Interaction] = []
    # Pool ONLY rating-retained IDs and discard review text at the loader boundary.
    # This keeps the large Yelp input practical without keeping its JSON objects.
    users: dict[str, str] = {}
    items: dict[str, str] = {}
    for row in rows:
        row = Interaction(users.setdefault(row.user, row.user), items.setdefault(row.item, row.item),
                          row.rating, row.timestamp, row.order)
        if strategy == "none":
            retained.append(row)
            continue
        key = (row.user, row.item)
        old = selected.get(key)
        if old is None:
            selected[key] = row
        else:
            counts["duplicates_removed"] += 1
            if (strategy == "earliest" and (row.timestamp, row.order) < (old.timestamp, old.order)
                    or strategy == "latest" and (row.timestamp, row.order) > (old.timestamp, old.order)):
                selected[key] = row
    if strategy != "none":
        retained = list(selected.values())
        retained.sort(key=lambda row: row.order)
    counts["after_dedup"] = len(retained)
    return retained


def iterative_k_core(rows: list[Interaction], user_core: int, item_core: int
                     ) -> tuple[list[Interaction], list[dict]]:
    rounds = []
    while rows:
        users = Counter(row.user for row in rows)
        items = Counter(row.item for row in rows)
        remaining = [row for row in rows
                     if users[row.user] >= user_core and items[row.item] >= item_core]
        rounds.append({"round": len(rounds) + 1, "before": len(rows), "after": len(remaining)})
        print(f"  Core round {len(rounds)}: {len(rows):,} -> {len(remaining):,}", flush=True)
        if len(remaining) == len(rows):
            assert min(users.values()) >= user_core and min(items.values()) >= item_core
            return rows, rounds
        rows = remaining
    return rows, rounds


def build_sequences(rows: list[Interaction]) -> dict[str, list[Interaction]]:
    sequences: dict[str, list[Interaction]] = {}
    for row in rows:
        sequences.setdefault(row.user, []).append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda row: (row.timestamp, row.order))
    return sequences


def remap_ids(rows: list[Interaction]) -> tuple[dict[str, int], dict[str, int]]:
    item2id: dict[str, int] = {}
    user2id: dict[str, int] = {}
    for row in rows:  # All preceding filters preserve surviving raw-file order.
        if row.item not in item2id:
            item2id[row.item] = len(item2id) + 1
        if row.user not in user2id:
            user2id[row.user] = len(user2id)
    return item2id, user2id


def split_sequences(sequences: dict[str, list[Interaction]], item2id: dict[str, int],
                    user2id: dict[str, int]) -> dict[str, list[list[int]]]:
    data: dict[str, list[list[int]]] = {"train": [], "val": [], "test": []}
    for user in user2id:
        sequence = [item2id[row.item] for row in sequences[user]]
        if len(sequence) < 4:
            raise ValueError("Each user needs at least four interactions for train/val/test splitting.")
        data["train"].append(sequence[:-2])
        data["val"].append([sequence[-2]])
        data["test"].append([sequence[-1]])
    return data


def compute_stats(rows: list[Interaction], data: dict) -> dict:
    users = Counter(row.user for row in rows)
    items = Counter(row.item for row in rows)
    lengths = list(users.values())
    return {
        "num_users": len(users), "num_items": len(items), "num_interactions": len(rows),
        "train_interactions": sum(map(len, data["train"])),
        "val_interactions": sum(map(len, data["val"])),
        "test_interactions": sum(map(len, data["test"])),
        "avg_sequence_length": len(rows) / len(users),
        "min_sequence_length": min(lengths), "max_sequence_length": max(lengths),
        "median_sequence_length": statistics.median(lengths),
        "min_user_interactions": min(lengths), "max_user_interactions": max(lengths),
        "min_item_interactions": min(items.values()), "max_item_interactions": max(items.values()),
        "sparsity": 1 - len(rows) / (len(users) * len(items)),
        "min_item_id": 1, "max_item_id": len(items), "padding_id": 0,
    }


def load_metadata(path: Path | None, item2id: dict[str, int]) -> tuple[dict, dict]:
    """Read optional item metadata without changing interaction sequences."""
    counts: Counter = Counter()
    id2item = {str(index): {"original_id": item} for item, index in item2id.items()}
    if path is None:
        return id2item, {"provided": False}
    # ASIN is useful even when the optional metadata record/title does not exist.
    for value in id2item.values():
        value["asin"] = value["original_id"]
    found = set()
    for _, record in iter_json_records(path, counts, literal_fallback=True):
        asin = clean_id(record.get("asin"))
        if asin is None:
            counts["invalid_asin"] += 1
            continue
        if asin not in item2id:
            continue
        counts["duplicate_retained_asin"] += asin in found
        found.add(asin)
        item = id2item[str(item2id[asin])]
        title = record.get("title")
        if "title" not in item and isinstance(title, str) and title.strip():
            item["title"] = title.strip()  # first nonempty title, independent of set order
    return id2item, {"provided": True, **dict(counts), "matched_items": len(found),
                     "items_without_meta": len(item2id) - len(found)}


def validate_dataset(data: dict, sequences: dict[str, list[Interaction]], item2id: dict,
                     user2id: dict, user_core: int, item_core: int) -> None:
    assert type(data) is dict and set(data) == {"train", "val", "test"}
    n_users, n_items = len(user2id), len(item2id)
    assert n_users > 0 and n_items > 0
    assert set(item2id.values()) == set(range(1, n_items + 1))
    assert set(user2id.values()) == set(range(n_users))
    assert all(type(split) is list and len(split) == n_users for split in data.values())
    counts: Counter = Counter()
    for user, index in user2id.items():
        train, val, test = (data[key][index] for key in ("train", "val", "test"))
        assert all(type(part) is list for part in (train, val, test))
        assert len(train) >= 2 and len(val) == len(test) == 1
        sequence = train + val + test
        assert all(type(item) is int and 0 < item <= n_items for item in sequence)
        assert len(sequence) >= user_core
        original = sequences[user]
        assert sequence == [item2id[row.item] for row in original]
        ordering = [(row.timestamp, row.order) for row in original]
        assert all(left <= right for left, right in zip(ordering, ordering[1:]))
        assert original[-3].timestamp <= original[-2].timestamp <= original[-1].timestamp
        counts.update(sequence)
    assert set(counts) == set(range(1, n_items + 1)) and 0 not in counts
    assert max(counts) == n_items and min(counts.values()) >= item_core
    print("Validation passed: IDs, splits, timestamps and interaction thresholds.", flush=True)


def print_summary(stats: dict) -> None:
    print("\n" + "=" * 44)
    print(f"Dataset: {stats['dataset']}")
    for key, label in (("num_users", "Users"), ("num_items", "Items"),
                       ("num_interactions", "Interactions"), ("min_sequence_length", "Min Seq Len"),
                       ("max_sequence_length", "Max Seq Len")):
        print(f"{label:20s} {stats[key]:14,}")
    print(f"Avg Seq Len:         {stats['avg_sequence_length']:14.4f}")
    print(f"Sparsity:            {stats['sparsity']:14.6%}")
    print("=" * 44, flush=True)


def check_output(output: Path, inputs: list[Path], overwrite: bool) -> None:
    protected = (ROOT / "datasets/data").resolve()
    resolved = output.resolve()
    if resolved == protected or protected in resolved.parents:
        raise ValueError("datasets/data is protected; choose datasets/generated/<dataset> instead.")
    for name in ARTIFACTS:
        target = resolved / name
        if target.is_symlink() or target.resolve() in inputs:
            raise ValueError(f"Refusing to write an input file or output symlink: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; choose another --output or use --overwrite.")


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_dataset(args: argparse.Namespace, data: dict, item2id: dict, id2item: dict,
                 user2id: dict, stats: dict, sequences: dict) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-staging-", dir=args.output) as temp:
        staging = Path(temp)
        with (staging / "dataset.pkl").open("wb") as stream:
            pickle.dump(data, stream, protocol=4)
        with (staging / "dataset.pkl").open("rb") as stream:
            loaded = pickle.load(stream)
        validate_dataset(loaded, sequences, item2id, user2id, args.user_core, args.item_core)
        for name, value in (("item2id.json", item2id), ("id2item.json", id2item),
                            ("user2id.json", user2id), ("stats.json", stats)):
            write_json(staging / name, value)
        for name in ARTIFACTS:
            os.replace(staging / name, args.output / name)
    print(f"Saved: {args.output}", flush=True)


def rating_argument(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        rating = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use none or a finite threshold, e.g. 3, 4, 5.") from error
    if not math.isfinite(rating):
        raise argparse.ArgumentTypeError("Rating threshold must be finite.")
    return rating


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str.lower, choices=DATASETS, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Raw reviews JSON/JSON.GZ or MovieLens u.data")
    parser.add_argument("--meta", type=Path, help="Optional Amazon metadata; never required for sequences")
    parser.add_argument("--output", type=Path, help="Default: <repo>/datasets/generated/<dataset>")
    parser.add_argument("--min_rating", type=rating_argument, default=4.0, metavar="none|3|4|5",
                        help="Default 4; missing/invalid ratings are retained")
    parser.add_argument("--dedup", choices=("earliest", "latest", "none"), default="earliest")
    parser.add_argument("--user_core", type=int, default=5)
    parser.add_argument("--item_core", type=int, default=5)
    parser.add_argument("--start_date", type=date_argument, help="Optional inclusive UTC start date, YYYY-MM-DD")
    parser.add_argument("--end_date", type=date_argument, help="Optional inclusive UTC end date, YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.user_core < 4 or args.item_core < 1:
        parser.error("--user_core must be >=4 and --item_core must be >=1.")
    if args.meta is not None and args.dataset not in AMAZON:
        parser.error("--meta is only supported for the four Amazon datasets.")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start_date must not be later than --end_date.")
    if sys.flags.optimize:
        parser.error("Do not use python -O: integrity assertions must be enabled.")
    for path in (args.input, args.meta):
        if path is not None and not path.is_file():
            parser.error(f"Input file not found: {path}")
    args.output = (args.output or ROOT / "datasets/generated" / args.dataset).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    check_output(args.output, [p.resolve() for p in (args.input, args.meta) if p is not None], args.overwrite)
    print(f"Dataset={args.dataset}, min_rating={args.min_rating} (missing kept), "
          f"dedup={args.dedup}, user_core={args.user_core}, item_core={args.item_core}", flush=True)
    if args.start_date or args.end_date:
        print(f"UTC date window (inclusive days): {args.start_date or 'unbounded'} "
              f"through {args.end_date or 'unbounded'}", flush=True)
    counts: Counter = Counter()
    raw_rows = filter_dates(load_raw_data(args.dataset, args.input, counts), args.start_date, args.end_date, counts)
    rows = deduplicate(filter_rating(raw_rows, args.min_rating, counts),
                       args.dedup, counts)
    print("Raw/rating/dedup counts:", json.dumps(counts, sort_keys=True), flush=True)
    rows, rounds = iterative_k_core(rows, args.user_core, args.item_core)
    if not rows:
        raise ValueError("No interactions survive filtering/core; no dataset was written. Check input/settings.")
    sequences = build_sequences(rows)
    item2id, user2id = remap_ids(rows)
    data = split_sequences(sequences, item2id, user2id)
    stats = compute_stats(rows, data)
    stats.update({"dataset": args.dataset, "min_rating": args.min_rating, "dedup": args.dedup,
                  "user_core": args.user_core, "item_core": args.item_core,
                  "protocol": ("date_then_" if args.start_date or args.end_date else "")
                              + "rating_then_dedup_then_iterative_core_then_leave_last_two",
                  "start_date": args.start_date, "end_date": args.end_date,
                  "date_bounds": "inclusive UTC calendar dates; no filtering when both are null",
                  "missing_rating_policy": "keep", "invalid_rating_policy": "treat_as_missing",
                  "id_order": "first_surviving_raw_order", "timestamp_tie_break": "raw_order",
                  "date_timezone": "UTC", "user_id_base": 0, "parsing_and_filtering": dict(counts),
                  "core_rounds": rounds, "sparsity_definition": "1 - interactions / (users * items)",
                  "input": {name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
                            for name in ("input", "meta") if (path := getattr(args, name)) is not None}})
    print_summary(stats)
    id2item, metadata_counts = load_metadata(args.meta, item2id)
    stats["metadata"] = metadata_counts
    save_dataset(args, data, item2id, id2item, user2id, stats, sequences)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, AssertionError, EOFError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
