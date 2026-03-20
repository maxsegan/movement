#!/usr/bin/env python3
"""
Prepare MoveNet-332 dataset for HuggingFace upload.

Iterates all processed Kinetics-700 NPZ + description pairs, computes joint angles,
assigns train/val splits, and writes Parquet shards.

Usage:
    python scripts/prepare_hf_dataset.py
    python scripts/prepare_hf_dataset.py --workers 16 --shard-size 1500000000
"""

import argparse
import hashlib
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path so we can import from training/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.kinetics_dataset import pose3d_to_joint_angles, joint_angles_to_sincos, _parse_description


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSE_DIR = PROJECT_ROOT / "data" / "kinetics_processed"
DESC_DIR = PROJECT_ROOT / "data" / "kinetics_full_output" / "descriptions"
OUTPUT_DIR = PROJECT_ROOT / "data" / "hf_dataset"
VAL_SPLIT = 0.02  # 2% val, matching KineticsPoseDataset


def _deterministic_hash(text: str) -> float:
    """Stable hash to split train/val without storing manifest (matches training code)."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _compress_array(arr: np.ndarray) -> bytes:
    """Compress a numpy array with zlib for Parquet storage."""
    return zlib.compress(arr.tobytes(), level=6)


def _parse_clip_id(clip_id: str) -> Tuple[str, str, str]:
    """Parse youtube_id, time_start, time_end from clip_id like 'a4UOhPiV4QE_000675_000685'."""
    parts = clip_id.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return clip_id, "", ""


# ---------------------------------------------------------------------------
# Per-clip processing (runs in worker processes)
# ---------------------------------------------------------------------------

def _process_clip(args: Tuple[Path, Path, str]) -> Optional[Dict]:
    """Process a single clip: load NPZ, compute joint angles, return row dict."""
    npz_path, desc_path, action_class = args
    clip_id = npz_path.stem

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  SKIP (npz load error): {clip_id}: {e}")
        return None

    pose3d = data["pose3d"].astype(np.float32)  # [F, 17, 3]
    num_frames = pose3d.shape[0]
    if num_frames == 0:
        return None

    # Compute joint angles for all frames, stored as sin/cos pairs to avoid
    # atan2 wraparound discontinuities. Values are naturally in [-1, 1].
    joint_angles_raw = np.stack(
        [pose3d_to_joint_angles(pose3d[t]) for t in range(num_frames)],
        axis=0,
    ).astype(np.float32)  # [F, 22]
    joint_angles = joint_angles_to_sincos(joint_angles_raw)  # [F, 44]

    # Parse metadata
    meta = data["meta"].astype(np.float32)  # [fps, num_video_frames, width, height]
    fps = float(meta[0])
    video_width = int(meta[2])
    video_height = int(meta[3])

    # Parse quality / hard cuts
    quality = float(data["quality"][0]) if "quality" in data else 0.0
    has_hard_cuts = bool(data["has_hard_cuts"][0]) if "has_hard_cuts" in data else False

    # Parse description
    instruction = _parse_description(desc_path)

    # Parse clip ID components
    youtube_id, time_start, time_end = _parse_clip_id(clip_id)

    # Train/val split
    hval = _deterministic_hash(clip_id)
    split = "val" if hval < VAL_SPLIT else "train"

    # Compress arrays
    return {
        "clip_id": clip_id,
        "action_class": action_class,
        "youtube_id": youtube_id,
        "time_start": time_start,
        "time_end": time_end,
        "split": split,
        "instruction": instruction,
        "fps": fps,
        "num_pose_frames": num_frames,
        "video_width": video_width,
        "video_height": video_height,
        "pose3d": _compress_array(pose3d),
        "keypoints2d": _compress_array(data["keypoints2d"].astype(np.float32)),
        "scores2d": _compress_array(data["scores2d"].astype(np.float32)),
        "bboxes": _compress_array(data["bboxes"].astype(np.float32)),
        "joint_angles": _compress_array(joint_angles),
        "frame_indices": _compress_array(data["indices"].astype(np.int32)),
        "tracking_confidence": _compress_array(
            data["tracking_confidence"].astype(np.float32)
            if "tracking_confidence" in data
            else np.ones(num_frames, dtype=np.float32)
        ),
        "has_hard_cuts": has_hard_cuts,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

PARQUET_SCHEMA = pa.schema([
    ("clip_id", pa.string()),
    ("action_class", pa.string()),
    ("youtube_id", pa.string()),
    ("time_start", pa.string()),
    ("time_end", pa.string()),
    ("split", pa.string()),
    ("instruction", pa.string()),
    ("fps", pa.float32()),
    ("num_pose_frames", pa.int32()),
    ("video_width", pa.int32()),
    ("video_height", pa.int32()),
    ("pose3d", pa.binary()),
    ("keypoints2d", pa.binary()),
    ("scores2d", pa.binary()),
    ("bboxes", pa.binary()),
    ("joint_angles", pa.binary()),
    ("frame_indices", pa.binary()),
    ("tracking_confidence", pa.binary()),
    ("has_hard_cuts", pa.bool_()),
    ("quality", pa.float32()),
])


def _rows_to_table(rows: List[Dict]) -> pa.Table:
    """Convert list of row dicts to a PyArrow table."""
    columns = {field.name: [] for field in PARQUET_SCHEMA}
    for row in rows:
        for col in columns:
            columns[col].append(row[col])

    arrays = []
    for field in PARQUET_SCHEMA:
        arrays.append(pa.array(columns[field.name], type=field.type))
    return pa.table(arrays, schema=PARQUET_SCHEMA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_clip_pairs() -> List[Tuple[Path, Path, str]]:
    """Find all (npz_path, desc_path, action_class) triples."""
    pairs = []
    for npz_path in sorted(POSE_DIR.rglob("*.npz")):
        action_class = npz_path.parent.name
        clip_id = npz_path.stem
        desc_path = DESC_DIR / action_class / f"{clip_id}.txt"
        if desc_path.exists():
            pairs.append((npz_path, desc_path, action_class))
    return pairs


def write_shards(
    rows: List[Dict],
    split: str,
    output_dir: Path,
    shard_size_bytes: int,
):
    """Write rows as Parquet shards, splitting by approximate target shard size."""
    if not rows:
        return

    # Sort rows by action_class for better compression
    rows.sort(key=lambda r: (r["action_class"], r["clip_id"]))

    # Estimate bytes per row from first batch
    sample_table = _rows_to_table(rows[:100])
    buf = pa.BufferOutputStream()
    pq.write_table(sample_table, buf, compression="snappy")
    bytes_per_row = buf.getvalue().size / min(100, len(rows))

    rows_per_shard = max(1, int(shard_size_bytes / bytes_per_row))
    num_shards = max(1, (len(rows) + rows_per_shard - 1) // rows_per_shard)

    print(f"  {split}: {len(rows)} rows, ~{bytes_per_row:.0f} bytes/row, "
          f"{num_shards} shards of ~{rows_per_shard} rows")

    for shard_idx in range(num_shards):
        start = shard_idx * rows_per_shard
        end = min(start + rows_per_shard, len(rows))
        shard_rows = rows[start:end]

        table = _rows_to_table(shard_rows)
        shard_name = f"{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        out_path = output_dir / shard_name
        pq.write_table(table, out_path, compression="snappy")

        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"    Wrote {shard_name} ({len(shard_rows)} rows, {file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare MoveNet-332 HuggingFace dataset")
    parser.add_argument("--workers", type=int, default=32, help="Number of worker processes")
    parser.add_argument("--shard-size", type=int, default=2_000_000_000,
                        help="Target shard size in bytes (default: 2GB)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of clips to process (for testing)")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find all clip pairs
    print("Finding clip pairs...")
    pairs = find_clip_pairs()
    print(f"Found {len(pairs)} clip pairs")

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs)} clips")

    # 2. Process clips with multiprocessing
    print(f"Processing clips with {args.workers} workers...")
    with Pool(processes=args.workers) as pool:
        results = []
        total = len(pairs)
        for i, result in enumerate(pool.imap_unordered(_process_clip, pairs, chunksize=256)):
            if result is not None:
                results.append(result)
            if (i + 1) % 10000 == 0 or (i + 1) == total:
                print(f"  Processed {i + 1}/{total} clips ({len(results)} valid)")

    print(f"Total valid clips: {len(results)}")

    # 3. Split into train/val
    train_rows = [r for r in results if r["split"] == "train"]
    val_rows = [r for r in results if r["split"] == "val"]
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    # 4. Write shards
    print("Writing train shards...")
    write_shards(train_rows, "train", output_dir, args.shard_size)

    print("Writing val shards...")
    write_shards(val_rows, "val", output_dir, args.shard_size)

    # 5. Print summary
    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.parquet"))
    print(f"\nDone! Total Parquet size: {total_bytes / (1024**3):.2f} GB")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
