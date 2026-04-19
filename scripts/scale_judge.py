#!/usr/bin/env python3
"""
Sharded, batched 235B VLM judge over a clip list.

Designed for scaling VLM judgment to 10k-332k clips. Uses:
  - ThreadPoolExecutor with N concurrent in-flight VLM requests
  - Parallel frame decoding (cv2) + overlay rendering per clip
  - Stable shard slice K/N (deterministic partition of clip list)
  - Resumable: skips clips already in the output file

Example:
  VLM_SERVER_URL=http://localhost:8000/v1 \\
  python3.12 scripts/scale_judge.py \\
    --clip-list tests/stratified_10k.txt \\
    --out tests/judge_strat10k_s1.json \\
    --shard 1/1 --concurrency 8
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow
import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.render_skeleton_overlay import draw_skeleton
from scripts.vlm_client import vlm_judge_server_batch

VIDEO_DIR = REPO / "data" / "kinetics_videos_full"
DATA_DIR = REPO / "data" / "movenet-332"


def load_rows(clip_ids):
    """Single pass over all parquet files; keeps only rows in clip_ids."""
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet in {DATA_DIR}")
    want = set(clip_ids)
    rows = {}
    for pf in files:
        t = pq.read_table(pf)
        cid_col = t.column("clip_id").to_pylist()
        hit = [i for i, c in enumerate(cid_col) if c in want]
        if not hit:
            continue
        for i in hit:
            rows[cid_col[i]] = {col: t.column(col)[i].as_py()
                                for col in t.column_names}
        if len(rows) == len(want):
            break
    return rows


def build_request(row, n_frames=6):
    """CPU work: decode frames + build pil overlay images."""
    F = row["num_pose_frames"]
    kp2d = np.frombuffer(zlib.decompress(row["keypoints2d"]),
                         dtype=np.float32).reshape(F, 17, 2)
    scores = np.frombuffer(zlib.decompress(row["scores2d"]),
                           dtype=np.float32).reshape(F, 17)
    fi = np.frombuffer(zlib.decompress(row["frame_indices"]),
                       dtype=np.int32).reshape(F)
    cid = row["clip_id"]
    vp = VIDEO_DIR / f"{cid}.mp4"
    if not vp.exists():
        return {"id": cid, "error": "no_video"}
    sel = np.linspace(0, F - 1, n_frames).astype(int)
    wanted_src = fi[sel].astype(np.int64)
    cap = cv2.VideoCapture(str(vp))
    imgs = []
    for si in wanted_src:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(si))
        ok, fr = cap.read()
        if ok:
            imgs.append(fr)
    cap.release()
    if len(imgs) < n_frames:
        return {"id": cid, "error": f"short_read:{len(imgs)}"}
    pil_imgs = []
    for i, si in enumerate(sel):
        overlay = draw_skeleton(imgs[i], kp2d[si], scores[si])
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_imgs.append(Image.fromarray(rgb))
    return {
        "id": cid,
        "images": pil_imgs,
        "instruction": row["instruction"],
        "action_class": row["action_class"],
    }


def shard_slice(items, k, n):
    """Deterministic 1-based shard slice: keep i-th item iff i % n == (k-1)."""
    return [c for i, c in enumerate(items) if i % n == (k - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-list", required=True,
                    help="Path to file with one clip_id per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="1/1", help="K/N, 1-based")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="In-flight VLM requests")
    ap.add_argument("--prep-workers", type=int, default=4,
                    help="CPU workers for frame decode + overlay")
    ap.add_argument("--chunk-size", type=int, default=32,
                    help="Clips per prep-then-send cycle")
    ap.add_argument("--checkpoint-every", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--server",
                    default=os.environ.get("VLM_SERVER_URL",
                                           "http://localhost:8000/v1"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    k, n = [int(x) for x in args.shard.split("/")]

    with open(args.clip_list) as f:
        all_cids = [x.strip() for x in f if x.strip()]
    my_cids = shard_slice(all_cids, k, n)
    if args.limit:
        my_cids = my_cids[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.resume and out_path.exists():
        try:
            with open(out_path) as f:
                for r in json.load(f)["results"]:
                    existing[r["clip_id"]] = r
        except Exception:
            pass
    todo = [c for c in my_cids if c not in existing]
    print(f"Shard {k}/{n}: total={len(my_cids)} done={len(existing)} "
          f"todo={len(todo)} concurrency={args.concurrency}")
    if not todo:
        print("Nothing to do.")
        return

    print("Loading parquet rows...")
    t_load = time.time()
    rows = load_rows(todo)
    print(f"  {len(rows)} rows in {time.time()-t_load:.1f}s")

    results = list(existing.values())
    t0 = time.time()
    done = 0

    def prep_one(cid):
        if cid not in rows:
            return {"id": cid, "error": "no_row"}
        try:
            return build_request(rows[cid])
        except Exception as ex:
            return {"id": cid,
                    "error": f"{type(ex).__name__}: {str(ex)[:100]}"}

    for chunk_start in range(0, len(todo), args.chunk_size):
        batch_cids = todo[chunk_start:chunk_start + args.chunk_size]
        with ThreadPoolExecutor(max_workers=args.prep_workers) as pex:
            reqs = list(pex.map(prep_one, batch_cids))
        good = [r for r in reqs if "error" not in r]
        errs = [r for r in reqs if "error" in r]
        for e in errs:
            results.append({"clip_id": e["id"], "error": e["error"]})
        if good:
            batch_results = vlm_judge_server_batch(
                good, concurrency=args.concurrency,
                server_url=args.server)
            for r in batch_results:
                r["clip_id"] = r.pop("id")
                results.append(r)
        done += len(reqs)
        dt = time.time() - t0
        rate = done / max(dt, 0.1) * 60
        ok_so_far = [r for r in results if r.get("tracking_good") is not None]
        vg = sum(1 for r in ok_so_far
                 if r.get("tracking_good") and r.get("motion_matches"))
        print(f"  [{done}/{len(todo)}] rate={rate:.1f}/min "
              f"VLM-good={vg}/{len(ok_so_far)}")
        if done >= args.checkpoint_every and (
                done % args.checkpoint_every < args.chunk_size):
            with open(out_path, "w") as f:
                json.dump({"n": len(results), "results": results}, f)

    with open(out_path, "w") as f:
        json.dump({"n": len(results), "results": results}, f)

    ok = [r for r in results if r.get("tracking_good") is not None]
    vg = sum(1 for r in ok
             if r.get("tracking_good") and r.get("motion_matches"))
    total_dt = time.time() - t0
    print(f"\nFinal: n={len(ok)}/{len(results)}  VLM-good={vg} "
          f"({100*vg/max(len(ok),1):.1f}%)  "
          f"wall={total_dt:.1f}s rate={len(todo)/max(total_dt,0.1)*60:.1f}/min")


if __name__ == "__main__":
    main()
