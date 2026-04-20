#!/usr/bin/env python3
"""
Run programmatic quality checks over all clips in the movenet-332 parquet,
join with VLM-judge results, and emit a combined verdict file with strict
(prog AND VLM-good) computed.

CPU-only; parallelizes across workers by parquet shard.

Usage:
  python3.12 scripts/prog_post_pass.py \\
    --vlm tests/scale_strat332k.json \\
    --out tests/scale_strat332k_combined.json \\
    --workers 8
"""
import argparse
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

DATA_DIR = REPO / "data" / "movenet-332"


def prog_verdict(row):
    """Compute prog issues for a single decompressed parquet row."""
    from validate_dataset_quality import (check_bbox_continuity,
                                           check_bbox_size_stability,
                                           check_pose_continuity,
                                           check_confidence_scores,
                                           check_bbox_iou_continuity,
                                           check_motion_magnitude,
                                           check_bbox_area)
    F = row["num_pose_frames"]
    kp2d = np.frombuffer(zlib.decompress(row["keypoints2d"]),
                         dtype=np.float32).reshape(F, 17, 2)
    scores = np.frombuffer(zlib.decompress(row["scores2d"]),
                           dtype=np.float32).reshape(F, 17)
    bboxes = np.frombuffer(zlib.decompress(row["bboxes"]),
                           dtype=np.float32).reshape(F, 4)
    pose3d = np.frombuffer(zlib.decompress(row["pose3d"]),
                           dtype=np.float32).reshape(F, 17, 3)
    fps = float(row["fps"])
    w = int(row["video_width"])
    h = int(row["video_height"])
    has_hc = bool(row["has_hard_cuts"])

    issues = []
    if has_hc:
        issues.append("HARD_CUTS")
    if F < 20:
        issues.append(f"TOO_SHORT_{F}")
    bc = check_bbox_continuity(bboxes)
    if bc["jump_ratio"] > 0.05:
        issues.append(f"BBOX_JUMPS_{bc['num_jumps']}")
    ss = check_bbox_size_stability(bboxes)
    if ss["spike_ratio"] > 0.03:
        issues.append(f"SIZE_SPIKES_{ss['num_size_spikes']}")
    pc = check_pose_continuity(pose3d, bboxes, scores)
    if pc["suspicious_ratio"] > 0.05:
        issues.append(f"POSE_JUMPS_{pc['num_suspicious_jumps']}")
    cf = check_confidence_scores(scores)
    if cf["low_confidence_ratio"] > 0.4:
        issues.append(f"LOW_CONFIDENCE_{int(cf['low_confidence_ratio']*100)}%")
    iou = check_bbox_iou_continuity(bboxes)
    if iou["iou_break_ratio"] > 0.1:
        issues.append(f"IOU_BREAKS_{iou['iou_break_count']}")
    motion = check_motion_magnitude(pose3d, fps)
    if motion["peak_joint_speed_mps"] < 3.0:
        issues.append(f"LOW_MOTION_{motion['peak_joint_speed_mps']:.1f}mps")
    area = check_bbox_area(bboxes, w, h)
    if (area["bbox_area_median"] is not None
            and area["bbox_area_median"] < 0.08):
        issues.append(f"SMALL_BBOX_{int(area['bbox_area_median']*100)}%")
    return len(issues) == 0, issues


def process_shard(pf_str):
    """Worker: prog verdict for every clip in one parquet shard."""
    out = {}
    t = pq.read_table(pf_str)
    cols = t.column_names
    n = t.num_rows
    for i in range(n):
        row = {c: t.column(c)[i].as_py() for c in cols}
        try:
            good, issues = prog_verdict(row)
            out[row["clip_id"]] = {"prog_good": good, "issues": issues}
        except Exception as ex:
            out[row["clip_id"]] = {
                "prog_good": None,
                "issues": [f"ERR_{type(ex).__name__}"],
                "error": str(ex)[:120],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True,
                    help="VLM result JSON (from scale_judge)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    print(f"Loading VLM results from {args.vlm}...")
    with open(args.vlm) as f:
        vlm = {r["clip_id"]: r for r in json.load(f)["results"]}
    print(f"  {len(vlm)} VLM verdicts")

    shards = sorted(DATA_DIR.glob("train-*.parquet"))
    print(f"Processing {len(shards)} parquet shards on {args.workers} workers...")
    prog = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_shard, str(p)) for p in shards]
        for i, f in enumerate(as_completed(futs), 1):
            d = f.result()
            prog.update(d)
            print(f"  shard {i}/{len(shards)}  +{len(d)} clips  total={len(prog)}",
                  flush=True)

    print("Joining prog + VLM...")
    combined = []
    for cid, p in prog.items():
        v = vlm.get(cid, {})
        vg_t = bool(v.get("tracking_good"))
        vg_m = bool(v.get("motion_matches"))
        vlm_good = vg_t and vg_m
        row = {
            "clip_id": cid,
            "prog_good": p["prog_good"],
            "prog_issues": p["issues"],
            "vlm_tracking_good": v.get("tracking_good"),
            "vlm_motion_matches": v.get("motion_matches"),
            "vlm_good": vlm_good,
            "strict": bool(p["prog_good"]) and vlm_good,
            "vlm_judged": "tracking_good" in v,
        }
        combined.append(row)

    # Write
    with open(args.out, "w") as f:
        json.dump({"n": len(combined), "results": combined}, f)

    judged = [r for r in combined if r["vlm_judged"]]
    strict = sum(1 for r in judged if r["strict"])
    vlm_good = sum(1 for r in judged if r["vlm_good"])
    prog_good = sum(1 for r in judged if r["prog_good"])
    err = sum(1 for r in combined if r["prog_good"] is None)
    total_prog_good = sum(1 for r in combined if r["prog_good"])

    print(f"\nTotal clips (parquet): {len(combined)}")
    print(f"  prog_good: {total_prog_good} ({100*total_prog_good/len(combined):.1f}%)")
    print(f"  prog errors: {err}")
    print(f"\nVLM-judged subset: {len(judged)}")
    if judged:
        print(f"  prog_good:  {prog_good} ({100*prog_good/len(judged):.1f}%)")
        print(f"  vlm_good:   {vlm_good} ({100*vlm_good/len(judged):.1f}%)")
        print(f"  STRICT:     {strict} ({100*strict/len(judged):.1f}%)")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
