#!/usr/bin/env python3
"""
Variant of scale_judge.py that reads overlay data from retrack NPZ files
(written by scripts/retrack_clips.py) instead of the original parquet.

Used for Phase 2 re-judge: vLLM evaluates the retracked skeleton overlays.

Same batched/sharded/resumable design as scale_judge.py.

Example:
  VLM_SERVER_URL=http://localhost:8000/v1 \\
  python3.12 scripts/scale_judge_npz.py \\
    --clip-list tests/phase2_retracked_ok.txt \\
    --npz-dirs tests/phase2_npz_s1 tests/phase2_npz_s2 tests/phase2_npz_s3 tests/phase2_npz_s4 \\
    --out tests/phase2_rejudge.json \\
    --concurrency 12 --chunk-size 48 --resume
"""
import argparse
import json
import os
import sys
import time
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


def load_metadata(clip_ids):
    """Load instruction + action_class from parquet for the given clips."""
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    want = set(clip_ids)
    out = {}
    for pf in files:
        t = pq.read_table(pf, columns=["clip_id", "instruction", "action_class"])
        cids = t.column("clip_id").to_pylist()
        ins = t.column("instruction").to_pylist()
        acs = t.column("action_class").to_pylist()
        for cid, ins_v, ac_v in zip(cids, ins, acs):
            if cid in want:
                out[cid] = {"instruction": ins_v, "action_class": ac_v}
        if len(out) == len(want):
            break
    return out


def find_npz(cid, npz_dirs):
    for d in npz_dirs:
        p = d / f"{cid}.npz"
        if p.exists():
            return p
    return None


def build_request(cid, npz_dirs, meta, n_frames=6):
    """Decode video frames + render retrack-skeleton overlays."""
    npz_path = find_npz(cid, npz_dirs)
    if npz_path is None:
        return {"id": cid, "error": "no_npz"}
    if cid not in meta:
        return {"id": cid, "error": "no_meta"}
    vp = VIDEO_DIR / f"{cid}.mp4"
    if not vp.exists():
        return {"id": cid, "error": "no_video"}

    d = np.load(npz_path)
    kp2d = d["keypoints2d"]
    scores = d["scores"]
    fi = d["frame_indices"]
    F = kp2d.shape[0]
    if F < n_frames:
        return {"id": cid, "error": f"too_short:{F}"}

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
        "instruction": meta[cid]["instruction"],
        "action_class": meta[cid]["action_class"],
    }


def shard_slice(items, k, n):
    return [c for i, c in enumerate(items) if i % n == (k - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-list", required=True)
    ap.add_argument("--npz-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--prep-workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=48)
    ap.add_argument("--checkpoint-every", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--server",
                    default=os.environ.get("VLM_SERVER_URL",
                                           "http://localhost:8000/v1"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    npz_dirs = [Path(d) if Path(d).is_absolute() else REPO / d
                for d in args.npz_dirs]
    for d in npz_dirs:
        if not d.exists():
            raise FileNotFoundError(d)

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

    print("Loading parquet metadata...")
    t_load = time.time()
    meta = load_metadata(todo)
    print(f"  {len(meta)} metadata rows in {time.time()-t_load:.1f}s")

    results = list(existing.values())
    t0 = time.time()
    done = 0

    def prep_one(cid):
        try:
            return build_request(cid, npz_dirs, meta)
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
