from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
INPUT_PATH = DATA_DIR / "df_gsi.csv"
OUTPUT_PATH = DATA_DIR / "df_gsi.csv"
MAX_SAMPLES = 4_000
RANDOM_SEED = 42


def parse_path(value: str) -> list[int]:
    path = ast.literal_eval(value)
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError(f"Invalid path: {value}")
    return [int(node_id) for node_id in path]


def main() -> None:
    labels_bot = torch.load(DATA_DIR / "labels_bot.pt", weights_only=False)
    source = pd.read_csv(INPUT_PATH)
    paths = source["full_path"].map(parse_path)
    human_mask = paths.map(
        lambda path: all(int(labels_bot[node_id]) == 0 for node_id in path)
    )
    human_paths = paths[human_mask]

    if len(human_paths) < MAX_SAMPLES:
        raise ValueError(
            f"Only {len(human_paths):,} human paths are available; "
            f"{MAX_SAMPLES:,} are required"
        )
    if len(human_paths) > MAX_SAMPLES:
        human_paths = human_paths.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED)

    output = pd.DataFrame(
        {
            "source_id": human_paths.map(lambda path: path[0]).to_numpy(),
            "target_id": human_paths.map(lambda path: path[-1]).to_numpy(),
            "mediator_ids": human_paths.map(
                lambda path: json.dumps(path[1:-1], separators=(",", ":"))
            ).to_numpy(),
            "full_path": human_paths.map(
                lambda path: json.dumps(path, separators=(",", ":"))
            ).to_numpy(),
            "hop_count": human_paths.map(lambda path: len(path) - 1).to_numpy(),
        }
    )
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(output):,} human-only GSI samples to {OUTPUT_PATH}")
    print(f"Hop counts: {output['hop_count'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()