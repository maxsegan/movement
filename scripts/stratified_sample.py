#!/usr/bin/env python3
"""
Build a stratified clip-id sample across action classes for scale validation.

Picks `--per-class` clips per action class (or all clips if the class has
fewer). Writes a clip-id list (one per line) sorted by class then by id, so
downstream sharding is stable.
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "movenet-332"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=14,
                    help="Clips per action class (cap)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    by_class = defaultdict(list)
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    for pf in files:
        t = pq.read_table(pf, columns=["clip_id", "action_class"])
        for cid, ac in zip(t.column("clip_id").to_pylist(),
                           t.column("action_class").to_pylist()):
            by_class[ac].append(cid)

    picks = []
    for ac, cids in sorted(by_class.items()):
        random.shuffle(cids)
        picks.extend(sorted(cids[:args.per_class]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picks) + "\n")
    print(f"Wrote {len(picks)} clip ids across {len(by_class)} classes to {out}")


if __name__ == "__main__":
    main()
