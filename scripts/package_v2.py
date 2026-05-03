#!/usr/bin/env python3
"""
Package the final composite verdict + retrack tracking data into a v2
dataset, in two formats:

  --format parquet  → HF-distributable parquet shards (one file per
                       original shard, filtered)
  --format npz      → per-clip NPZ + description.txt for training pipeline
                       (data/kinetics_v2_<filter>/<action_class>/...)

  --filter strict   → 164,390 clips (prog AND vlm)
  --filter vlm_good → 286,890 clips (vlm only — broader, lower bar)

For source=retrack rows we splice the retracked NPZ's tracking fields
(pose3d, keypoints2d, scores2d, bboxes, frame_indices, num_pose_frames,
fps, video_width, video_height, has_hard_cuts) into the parquet row,
preserving original metadata (action_class, instruction, youtube_id,
time_start, time_end, split). For source=original, the row is unchanged
except for the new v2_* quality columns.
"""
import argparse
import io
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "movenet-332"
NPZ_DIRS = [REPO / f"tests/phase2_npz_s{s}" for s in (1, 2, 3, 4)]


def find_npz(cid):
    for d in NPZ_DIRS:
        p = d / f"{cid}.npz"
        if p.exists():
            return p
    return None


def load_verdicts():
    with open(REPO / "tests/final_composite.json") as f:
        return {r["clip_id"]: r for r in json.load(f)["results"]}


def splice_retrack_into_row(orig_row, npz_path):
    """Replace tracking fields with retracked NPZ contents."""
    d = np.load(npz_path)
    F = int(d["pose3d"].shape[0])
    out = dict(orig_row)
    out["num_pose_frames"] = F
    out["fps"] = float(d["fps"])
    out["video_width"] = int(d["video_width"])
    out["video_height"] = int(d["video_height"])
    out["has_hard_cuts"] = bool(d["has_hard_cuts"])
    out["pose3d"] = zlib.compress(
        d["pose3d"].astype(np.float32).tobytes())
    out["keypoints2d"] = zlib.compress(
        d["keypoints2d"].astype(np.float32).tobytes())
    out["scores2d"] = zlib.compress(
        d["scores"].astype(np.float32).tobytes())
    out["bboxes"] = zlib.compress(
        d["bboxes"].astype(np.float32).tobytes())
    out["frame_indices"] = zlib.compress(
        d["frame_indices"].astype(np.int32).tobytes())
    # joint_angles: drop (recomputable from pose3d at training time)
    out["joint_angles"] = b""
    # tracking_confidence: per-frame mean across joints, zlib-compressed (F,)
    sc = d["scores"].astype(np.float32)
    per_frame_conf = np.nanmean(sc, axis=1).astype(np.float32)
    out["tracking_confidence"] = zlib.compress(per_frame_conf.tobytes())
    out["quality"] = float(orig_row.get("quality") or 0.0)
    return out


def process_parquet_shard(args):
    """Worker: read one parquet shard, filter+splice, write filtered v2 shard."""
    pf_in_str, out_dir, filter_name, verdicts = args
    pf_in = Path(pf_in_str)
    out_dir = Path(out_dir)
    t = pq.read_table(pf_in)
    cols = t.column_names

    n = t.num_rows
    keep_idx = []
    rows_out = []
    for i in range(n):
        cid = t.column("clip_id")[i].as_py()
        v = verdicts.get(cid)
        if v is None:
            continue
        if filter_name == "strict" and not v["strict"]:
            continue
        if filter_name == "vlm_good" and not v["vlm_good"]:
            continue
        row = {c: t.column(c)[i].as_py() for c in cols}
        if v["source"] == "retrack":
            npz = find_npz(cid)
            if npz is None:
                # Defensive fallback — should not happen
                v_eff = dict(v); v_eff["source"] = "original_fallback"
            else:
                row = splice_retrack_into_row(row, npz)
                v_eff = v
        else:
            v_eff = v
        # Add v2_* columns
        row["v2_source"] = v_eff["source"]
        row["v2_prog_good"] = bool(v_eff["prog_good"]) if v_eff["prog_good"] is not None else False
        row["v2_vlm_tracking_good"] = bool(v_eff["vlm_tracking_good"]) if v_eff["vlm_tracking_good"] is not None else False
        row["v2_vlm_motion_matches"] = bool(v_eff["vlm_motion_matches"]) if v_eff["vlm_motion_matches"] is not None else False
        row["v2_vlm_good"] = bool(v_eff["vlm_good"])
        row["v2_strict"] = bool(v_eff["strict"])
        row["v2_prog_issues"] = list(v_eff.get("prog_issues") or [])
        rows_out.append(row)

    if not rows_out:
        return {"shard": pf_in.name, "n": 0, "out": None}

    # Build pyarrow table from rows_out
    schema_fields = []
    src_schema = t.schema
    for f in src_schema:
        schema_fields.append(f)
    # New v2 columns
    schema_fields += [
        pa.field("v2_source", pa.string()),
        pa.field("v2_prog_good", pa.bool_()),
        pa.field("v2_vlm_tracking_good", pa.bool_()),
        pa.field("v2_vlm_motion_matches", pa.bool_()),
        pa.field("v2_vlm_good", pa.bool_()),
        pa.field("v2_strict", pa.bool_()),
        pa.field("v2_prog_issues", pa.list_(pa.string())),
    ]
    new_schema = pa.schema(schema_fields)

    arrays = []
    for f in new_schema:
        col = [r.get(f.name) for r in rows_out]
        arrays.append(pa.array(col, type=f.type))
    out_table = pa.Table.from_arrays(arrays, schema=new_schema)

    out_path = out_dir / pf_in.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path, compression="zstd")
    return {"shard": pf_in.name, "n": len(rows_out), "out": str(out_path)}


def main_parquet(args):
    verdicts = load_verdicts()
    out_dir = REPO / "data" / f"movement-{args.filter.replace('_','-')}-v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(DATA_DIR.glob("train-*.parquet"))
    print(f"Packaging {len(shards)} shards → {out_dir}  filter={args.filter}")
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_parquet_shard,
                          (str(s), str(out_dir), args.filter, verdicts))
                for s in shards]
        for f in as_completed(futs):
            r = f.result()
            print(f"  {r['shard']}: {r['n']} kept")
            total += r["n"]
    print(f"\nTotal kept: {total} clips → {out_dir}")


def write_npz_for_clip(cid, verdict, orig_row, npz_dir, desc_dir):
    """For training format. orig_row is None when source=retrack."""
    ac = verdict.get("action_class")  # may be None — caller should set
    if verdict["source"] == "retrack":
        src_npz = find_npz(cid)
        if src_npz is None:
            return False
        d = np.load(src_npz)
        pose3d = d["pose3d"].astype(np.float32)
        bboxes = d["bboxes"].astype(np.float32)
        indices = d["frame_indices"].astype(np.int32)
    else:
        F = int(orig_row["num_pose_frames"])
        pose3d = np.frombuffer(zlib.decompress(orig_row["pose3d"]),
                               dtype=np.float32).reshape(F, 17, 3)
        bboxes = np.frombuffer(zlib.decompress(orig_row["bboxes"]),
                               dtype=np.float32).reshape(F, 4)
        indices = np.frombuffer(zlib.decompress(orig_row["frame_indices"]),
                                dtype=np.int32).reshape(F)
    cls_dir = npz_dir / ac
    cls_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cls_dir / f"{cid}.npz",
                        pose3d=pose3d, bboxes=bboxes, indices=indices)
    cls_desc_dir = desc_dir / ac
    cls_desc_dir.mkdir(parents=True, exist_ok=True)
    instruction = orig_row.get("instruction") if orig_row else None
    if instruction is None:
        instruction = ""
    (cls_desc_dir / f"{cid}.txt").write_text(
        f"clip_id: {cid}\naction_class: {ac}\n\n"
        f"Description:\n{instruction}\n")
    return True


def process_npz_shard(args):
    """Worker: read one parquet shard, write per-clip NPZ + desc for matches."""
    pf_in_str, npz_root, desc_root, filter_name, verdicts = args
    pf_in = Path(pf_in_str)
    npz_root = Path(npz_root); desc_root = Path(desc_root)
    t = pq.read_table(pf_in)
    cols = t.column_names
    cids = t.column("clip_id").to_pylist()

    n_written = 0
    n_failed = 0
    for i, cid in enumerate(cids):
        v = verdicts.get(cid)
        if v is None:
            continue
        if filter_name == "strict" and not v["strict"]:
            continue
        if filter_name == "vlm_good" and not v["vlm_good"]:
            continue
        row = {c: t.column(c)[i].as_py() for c in cols}
        # Stash action_class into verdict
        v_eff = dict(v); v_eff["action_class"] = row["action_class"]
        try:
            ok = write_npz_for_clip(cid, v_eff, row, npz_root, desc_root)
            n_written += 1 if ok else 0
            n_failed += 0 if ok else 1
        except Exception as ex:
            n_failed += 1
            print(f"  ERR {cid}: {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    return {"shard": pf_in.name, "written": n_written, "failed": n_failed}


def main_npz(args):
    verdicts = load_verdicts()
    npz_root = REPO / "data" / f"kinetics_v2_{args.filter}"
    desc_root = REPO / "data" / f"kinetics_v2_{args.filter}_desc"
    npz_root.mkdir(parents=True, exist_ok=True)
    desc_root.mkdir(parents=True, exist_ok=True)
    shards = sorted(DATA_DIR.glob("train-*.parquet"))
    print(f"NPZ packaging {len(shards)} shards → {npz_root}  filter={args.filter}")
    total_w = 0; total_f = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_npz_shard,
                          (str(s), str(npz_root), str(desc_root),
                           args.filter, verdicts))
                for s in shards]
        for f in as_completed(futs):
            r = f.result()
            print(f"  {r['shard']}: written={r['written']} failed={r['failed']}")
            total_w += r["written"]; total_f += r["failed"]
    print(f"\nTotal: written={total_w} failed={total_f}")
    print(f"  Pose NPZs:    {npz_root}")
    print(f"  Descriptions: {desc_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["parquet", "npz"], required=True)
    ap.add_argument("--filter", choices=["strict", "vlm_good"], required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()
    if args.format == "parquet":
        main_parquet(args)
    else:
        main_npz(args)


if __name__ == "__main__":
    main()
