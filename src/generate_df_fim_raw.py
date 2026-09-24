from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "df_fim_raw.csv"
TEMP_PATH = DATA_DIR / "df_fim_raw.csv.tmp"
EDGE_TYPES = {0, 2, 3, 4}
MAX_SAMPLES = 4_000
RANDOM_SEED = 42
CHUNK_SIZE = 10_000


def main() -> None:
    edge_index = torch.load(DATA_DIR / "edge_index.pt", weights_only=False)
    edge_type = torch.load(DATA_DIR / "edge_type.pt", weights_only=False)
    labels_bot = torch.load(DATA_DIR / "labels_bot.pt", weights_only=False)
    features = torch.load(DATA_DIR / "features_pca256.pt", weights_only=False)

    if edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.shape[1] != edge_type.shape[0]:
        raise ValueError("edge_index and edge_type contain different numbers of edges")
    if features.ndim != 2 or features.shape[1] != 256:
        raise ValueError(f"Expected features shape [N, 256], got {tuple(features.shape)}")

    human_ids = set((labels_bot == 0).nonzero(as_tuple=True)[0].tolist())
    selected = [
        index
        for index, value in enumerate(edge_type.tolist())
        if value in EDGE_TYPES
        and int(edge_index[0, index]) in human_ids
        and int(edge_index[1, index]) in human_ids
    ]
    if len(selected) > MAX_SAMPLES:
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        sample_positions = torch.randperm(
            len(selected), generator=generator
        )[:MAX_SAMPLES].tolist()
        selected = [selected[position] for position in sample_positions]

    selected_edge_index = edge_index[:, selected]
    selected_edge_type = edge_type[selected]

    first_write = True
    for start in range(0, len(selected), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(selected))
        node_a = selected_edge_index[0, start:stop].tolist()
        node_b = selected_edge_index[1, start:stop].tolist()
        types = selected_edge_type[start:stop].tolist()

        rows = pd.DataFrame(
            {
                "node_A": node_a,
                "node_B": node_b,
                "node_A_profile": [
                    json.dumps(features[index].tolist(), separators=(",", ":"))
                    for index in node_a
                ],
                "node_B_profile": [
                    json.dumps(features[index].tolist(), separators=(",", ":"))
                    for index in node_b
                ],
                "edge_type": types,
            }
        )
        rows.to_csv(
            TEMP_PATH,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )
        first_write = False

    TEMP_PATH.replace(OUTPUT_PATH)
    print(f"Saved {len(selected):,} human-human rows to {OUTPUT_PATH}")
    print("Columns: node_A, node_B, node_A_profile, node_B_profile, edge_type")
    print(f"Included edge types: {sorted(EDGE_TYPES)}")


if __name__ == "__main__":
    main()