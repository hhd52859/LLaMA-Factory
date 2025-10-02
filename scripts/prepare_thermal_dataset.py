"""Convert thermal conversation data into the generic VLM fine-tuning schema.

This helper consumes raw JSON/JSONL annotations whose structure matches the
example provided by the user.  Every record is expected to look like::

    {
        "thermal": "000000086646.jpg",
        "conversations": [
            {"from": "human", "value": "<thermal>\nQuestion"},
            {"from": "gpt", "value": "Answer"}
        ]
    }

The script extracts (prompt, response) pairs from the conversation list and
emits three JSONL files compatible with :mod:`scripts.train_vlm`.  Each record
contains ``image``, ``instruction`` and ``output`` columns, with ``image``
storing paths resolved relative to ``--data_root``.  By default the converted
examples are randomly shuffled and split into ``train.jsonl``, ``validation.jsonl``
and ``test.jsonl`` according to an 80/10/10 ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, List, MutableMapping, Sequence

Record = MutableMapping[str, object]
Conversation = MutableMapping[str, object]


def _load_records(path: Path) -> List[Record]:
    """Load raw annotations from ``path``.

    Both JSON arrays and JSONL/NDJSON files are supported.  Empty lines are
    ignored for JSONL inputs.
    """

    if path.suffix.lower() in {".json", ".jsn"}:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must contain a list of objects.")
        return list(data)

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records: List[Record] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def _iter_pairs(conversations: Sequence[Conversation]) -> Iterator[tuple[str, str]]:
    """Yield successive (prompt, response) pairs from the conversation log."""

    i = 0
    while i < len(conversations):
        human = conversations[i]
        if human.get("from") != "human":
            i += 1
            continue

        try:
            assistant = conversations[i + 1]
        except IndexError:
            break

        if assistant.get("from") not in {"gpt", "assistant"}:
            i += 1
            continue

        human_value = str(human.get("value", "")).strip()
        assistant_value = str(assistant.get("value", "")).strip()

        if human_value and assistant_value:
            prompt = human_value.replace("<thermal>", "<image>")
            yield prompt, assistant_value

        i += 2


def _resolve_image_path(root: Path, entry_path: str) -> str:
    image_path = Path(entry_path)
    if not image_path.is_absolute():
        image_path = root / image_path
    return os.fspath(image_path)


def convert_dataset(records: Iterable[Record], data_root: Path) -> List[dict[str, str]]:
    """Convert raw thermal records into the generic VLM schema."""

    examples: List[dict[str, str]] = []
    for record in records:
        image_field = record.get("thermal")
        if not isinstance(image_field, str):
            raise ValueError("Each record must contain a 'thermal' string field.")

        conversations = record.get("conversations")
        if not isinstance(conversations, Sequence):
            raise ValueError("Each record must contain a 'conversations' list.")

        resolved_path = _resolve_image_path(data_root, image_field)
        for instruction, output in _iter_pairs(conversations):
            examples.append(
                {
                    "image": resolved_path,
                    "instruction": instruction,
                    "output": output,
                }
            )
    return examples


def dump_jsonl(records: Iterable[MutableMapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to the raw JSON/JSONL annotations.")
    parser.add_argument(
        "--data_root",
        required=True,
        type=Path,
        help="Directory that stores the thermal image files referenced by the annotations.",
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output",
        dest="output",
        type=Path,
        help="Base directory or file path prefix for the exported JSONL splits.",
    )
    output_group.add_argument(
        "--output_dir",
        dest="output",
        type=Path,
        help="Alias for --output maintained for backward compatibility.",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN_RATIO", "VAL_RATIO", "TEST_RATIO"),
        default=(0.8, 0.1, 0.1),
        help="Ratios for splitting the dataset into train/validation/test subsets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle the dataset before splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.input)
    examples = convert_dataset(records, args.data_root)

    if not examples:
        raise ValueError("No valid conversations found in the input annotations.")

    train_ratio, val_ratio, test_ratio = args.split
    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must be non-negative.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    total = len(examples)
    train_count = math.floor(total * train_ratio)
    val_count = math.floor(total * val_ratio)
    test_count = total - train_count - val_count

    train_examples = examples[:train_count]
    val_examples = examples[train_count : train_count + val_count]
    test_examples = examples[train_count + val_count :]

    output_dir = args.output
    if output_dir.suffix:
        # The user provided a concrete file path such as ``train.jsonl``.  We ignore the
        # specific stem and reuse the suffix for train/validation/test files placed in
        # the same parent directory to satisfy the "three-way split" requirement while
        # remaining compatible with existing invocation examples.
        suffix = output_dir.suffix
        base_dir = output_dir.parent
    else:
        suffix = ".jsonl"
        base_dir = output_dir

    dump_jsonl(train_examples, base_dir / f"train{suffix}")
    dump_jsonl(val_examples, base_dir / f"validation{suffix}")
    dump_jsonl(test_examples, base_dir / f"test{suffix}")


if __name__ == "__main__":
    main()
