#!/usr/bin/env python3
"""
Track B harness: re-track the 11 needs-fix clips from manual_labels_bucket1.json.

Pipeline per clip:
  1. Load video + metadata (instruction, action_class, fps, w, h) from parquet.
  2. Multi-frame YOLO on 5 candidate-selection frames (10/30/50/70/90%).
  3. NMS-merge candidates across frames into a shared pool; render numbered
     boxes on the "busiest" frame.
  4. Qwen3-VL-32B oracle picks the target subject (index).
  5. For every target-fps frame: run YOLO, greedy-match to anchor (IoU > 0.3);
     if no match, carry the last-known box forward (until end) with NaN score.
  6. ViTPose inference on tracked boxes → 2D COCO kpts.
  7. COCO → H36M, MotionAGFormer 3D lifting.
  8. Write NPZ (one per clip) and a JSON row compatible with
     tests.validate_dataset_quality programmatic_verdict.
  9. Grade: run programmatic_verdict on retracked rows, compare to original
     rows, print flip stats.

Trim variant: for clips labeled needs_trimming, detect hard-cut positions on
the FULL frame sequence and retrack only the longest clean subrange.

Usage:
  python3.12 scripts/retrack_clips.py --out-dir tests/retrack_out \
      --out-json tests/retrack_results.json
"""
import argparse
import json
import os
import re
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "MotionAGFormer"))

from data_prep.fast_video_loader import (probe_video_meta_fast,
                                         read_frames_batch_fast,
                                         sample_indices_for_fps)
from data_prep.vitpose import infer_sequence
from data_prep.keypoints import h36m_coco_format
from data_prep.pose3d import load_motionagformer_from_path, lift_sequence_to_3d
from data_prep.pipeline.pipeline import detect_hard_cuts

LABELS_PATH = REPO / "tests" / "manual_labels_bucket1.json"
VIDEO_DIR = REPO / "data" / "kinetics_videos_full"
DATA_DIR = REPO / "data" / "movenet-332"
MAGFORMER_CKPT = REPO / "models" / "motionagformer-b-h36m.pth.tr"

YOLO_WEIGHTS = "yolov8x.pt"
YOLO_CONF = 0.30
MAX_CANDIDATES = 6
TARGET_FPS = 20.0
IOU_TRACK_THR = 0.5
SIZE_RATIO_MAX = 2.5
MAX_CONSECUTIVE_MISSES = 3
MIN_VALID_FRAMES = 20


# ---------------------------------------------------------------------------
# Detection + candidate pool
# ---------------------------------------------------------------------------

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def yolo_detect(yolo, frame_rgb, conf=YOLO_CONF):
    res = yolo.predict(source=frame_rgb, classes=[0], conf=conf,
                       verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (res.boxes.xyxy.cpu().numpy().astype(np.float32),
            res.boxes.conf.cpu().numpy().astype(np.float32))


def build_candidate_pool(yolo, frames, frame_indices):
    """
    Run YOLO on each given frame, pool candidates, NMS-merge across frames.

    Returns:
      pool: list of {box: xyxy, conf: float, frame_idx: local index into frames
                     where this candidate was rendered}
      per_frame_counts: list of ints (detection count per sampled frame)
    """
    per_frame_counts = []
    pool = []  # each entry: (box, conf, local_frame_idx)
    for i, fr in enumerate(frames):
        boxes, confs = yolo_detect(yolo, fr)
        per_frame_counts.append(len(boxes))
        for b, c in zip(boxes, confs):
            merged = False
            for j, p in enumerate(pool):
                if box_iou(p["box"], b) > 0.5:
                    # keep higher-conf version, and prefer more-central frame
                    if c > p["conf"]:
                        pool[j] = {"box": b, "conf": float(c),
                                   "sampled_frame": i}
                    merged = True
                    break
            if not merged:
                pool.append({"box": b, "conf": float(c), "sampled_frame": i})
    # sort by confidence and cap
    pool.sort(key=lambda p: -p["conf"])
    pool = pool[:MAX_CANDIDATES]
    return pool, per_frame_counts


# ---------------------------------------------------------------------------
# Oracle (single numbered frame)
# ---------------------------------------------------------------------------

def draw_numbered(frame_rgb, boxes_with_labels):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame_rgb).copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    colors = [(255, 80, 80), (80, 200, 80), (80, 120, 255), (255, 200, 50),
              (200, 80, 255), (80, 220, 220)]
    for i, box in boxes_with_labels:
        c = colors[(i - 1) % len(colors)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=c, width=4)
        tx, ty = x1 + 4, y1 + 4
        bb = draw.textbbox((tx, ty), str(i), font=font)
        draw.rectangle([bb[0] - 4, bb[1] - 4, bb[2] + 4, bb[3] + 4], fill=c)
        draw.text((tx, ty), str(i), fill=(0, 0, 0), font=font)
    return img


def oracle_pick(vlm_model, vlm_proc, pil_img, n, instruction, action_class):
    prompt = (
        f"One video frame with {n} numbered bounding boxes, each outlining a person.\n"
        f"Action label: {action_class}\n"
        f"Instruction: \"{instruction[:250]}\"\n\n"
        f"TASK: Pick the SINGLE numbered person who is the main subject performing this action "
        f"or who is most likely to perform it in this clip. If multiple people appear similar, "
        f"prefer the largest / most centered / most engaged figure. You MUST return one number "
        f"from 1 to {n}.\n\n"
        f"Reply with ONLY this JSON: "
        f'{{"chosen_index": <int 1..{n}>, "reason": "<under 20 words>"}}'
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt},
    ]}]
    text = vlm_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = vlm_proc(text=[text], images=[pil_img], videos=None,
                      padding=True, return_tensors="pt").to(vlm_model.device)
    with torch.no_grad():
        gen_ids = vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    out = vlm_proc.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0].strip()
    chosen, reason = None, ""
    try:
        if "{" in out and "}" in out:
            j = json.loads(out[out.index("{"):out.rindex("}") + 1])
            chosen = int(j.get("chosen_index"))
            reason = str(j.get("reason", ""))[:160]
    except (json.JSONDecodeError, ValueError, TypeError):
        m = re.search(r'"chosen_index"\s*:\s*(\d+)', out)
        if m:
            chosen = int(m.group(1))
    if chosen is not None and not (1 <= chosen <= n):
        chosen = None
    return chosen, reason, out[:240]


# ---------------------------------------------------------------------------
# IoU forward/backward tracker
# ---------------------------------------------------------------------------

def interpolate_missing(boxes):
    """Linearly interpolate NaN rows in a (T,4) box sequence from nearest
    valid neighbors. Edge NaNs copy from closest valid frame."""
    T = len(boxes)
    valid = np.isfinite(boxes[:, 0])
    if not valid.any():
        return boxes
    idx = np.arange(T)
    out = boxes.copy()
    for k in range(4):
        out[:, k] = np.interp(idx, idx[valid], boxes[valid, k])
    return out


def _match_next(prev, det_boxes, iou_thr, size_ratio_max):
    if len(det_boxes) == 0:
        return None
    ious = np.array([box_iou(prev, b) for b in det_boxes])
    best = int(np.argmax(ious))
    if ious[best] < iou_thr:
        return None
    prev_area = max(1.0, (prev[2] - prev[0]) * (prev[3] - prev[1]))
    cand = det_boxes[best]
    cand_area = max(1.0, (cand[2] - cand[0]) * (cand[3] - cand[1]))
    ratio = max(cand_area / prev_area, prev_area / cand_area)
    if ratio > size_ratio_max:
        return None
    return cand


def iou_track(yolo, frames, anchor_frame_idx, anchor_box,
              iou_thr=IOU_TRACK_THR, size_ratio_max=SIZE_RATIO_MAX,
              max_misses=MAX_CONSECUTIVE_MISSES):
    """
    Stickier forward/backward tracker. Breaks after max_misses consecutive
    frames without a high-IoU, similar-size match. Returns (boxes, forward_stop,
    backward_stop) where the contiguous valid run is
    [backward_stop+1 : forward_stop+1] and always contains anchor.
    """
    T = len(frames)
    boxes = np.full((T, 4), np.nan, dtype=np.float32)
    boxes[anchor_frame_idx] = anchor_box

    # Forward
    prev = anchor_box
    misses = 0
    last_valid = anchor_frame_idx
    for t in range(anchor_frame_idx + 1, T):
        det, _ = yolo_detect(yolo, frames[t])
        m = _match_next(prev, det, iou_thr, size_ratio_max)
        if m is None:
            misses += 1
            if misses > max_misses:
                break
            continue
        boxes[t] = m
        prev = m
        last_valid = t
        misses = 0
    forward_stop = last_valid

    # Backward
    prev = anchor_box
    misses = 0
    first_valid = anchor_frame_idx
    for t in range(anchor_frame_idx - 1, -1, -1):
        det, _ = yolo_detect(yolo, frames[t])
        m = _match_next(prev, det, iou_thr, size_ratio_max)
        if m is None:
            misses += 1
            if misses > max_misses:
                break
            continue
        boxes[t] = m
        prev = m
        first_valid = t
        misses = 0
    backward_stop = first_valid

    return boxes, backward_stop, forward_stop


def trim_to_valid_run(frames, boxes, start, end):
    """Trim frames + boxes to [start:end+1] inclusive. Interpolate remaining
    interior NaNs (those from miss-tolerance gaps)."""
    sub_frames = frames[start:end + 1]
    sub_boxes = boxes[start:end + 1].copy()
    sub_boxes = interpolate_missing(sub_boxes)
    return sub_frames, sub_boxes, start, end


# ---------------------------------------------------------------------------
# Trim-subrange detection
# ---------------------------------------------------------------------------

def choose_trim_range(frames):
    """Return (start, end) exclusive for the longest hard-cut-free subrange."""
    hc = detect_hard_cuts(frames)
    if not hc:
        return 0, len(frames)
    edges = [0] + list(sorted(set(hc))) + [len(frames)]
    best_s, best_e, best_len = 0, len(frames), 0
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        if (e - s) > best_len:
            best_s, best_e, best_len = s, e, e - s
    return best_s, best_e


# ---------------------------------------------------------------------------
# Pose + 3D
# ---------------------------------------------------------------------------

def run_vitpose(vitpose_proc, vitpose_model, device, frames, boxes_xyxy):
    idxs = np.arange(len(frames), dtype=np.int64)
    kpts, scrs = infer_sequence(
        raw_video_path="",
        image_processor=vitpose_proc,
        model=vitpose_model,
        device=torch.device(device),
        idxs=idxs,
        boxes_xyxy=boxes_xyxy,
        frames=frames,
        batch_size=16,
    )
    # kpts: (1, T, 17, 2); scrs: (1, T, 17) -- COCO order
    return kpts[0], scrs[0]


def _median_filter_3d(y3d, window=3):
    """3-point (temporal) median filter on (F, J, 3) — robust to single-frame
    spikes without eroding real motion. Edges handled by reflection."""
    if y3d.shape[0] < window:
        return y3d
    w = window // 2
    pad = np.concatenate([y3d[1:w + 1][::-1], y3d, y3d[-w - 1:-1][::-1]], axis=0)
    # Stack windowed slices and take median along axis 0
    stacks = np.stack([pad[i:i + y3d.shape[0]] for i in range(window)], axis=0)
    return np.median(stacks, axis=0)


def lift_3d(seq_kpts_coco, seq_scores_coco, width, height, model_3d, device,
            smooth=False):
    """Convert COCO 2D kpts → H36M → 3D lifted (with optional median smooth)."""
    h36m_kpts, h36m_scores, _ = h36m_coco_format(
        seq_kpts_coco[None, ...], seq_scores_coco[None, ...])
    if h36m_kpts.shape[0] == 0:
        raise RuntimeError("h36m_coco_format produced no sequences")
    seq_k = h36m_kpts[0]  # (F, 17, 2)
    seq_s = h36m_scores[0]
    y3d = lift_sequence_to_3d(
        seq_k[None, ...], seq_s[None, ...], width, height, model_3d, device)
    if smooth:
        y3d = _median_filter_3d(y3d, window=3)
    return y3d, seq_k, seq_s


# ---------------------------------------------------------------------------
# Validation row assembly + programmatic verdict
# ---------------------------------------------------------------------------

def build_row(clip_id, fps, width, height, y3d, kp2d, scores, bboxes,
              instruction, action_class, hard_cuts):
    F = y3d.shape[0]
    return {
        "clip_id": clip_id,
        "instruction": instruction,
        "action_class": action_class,
        "fps": fps,
        "video_width": width,
        "video_height": height,
        "has_hard_cuts": len(hard_cuts) > 0,
        "hard_cut_frames": list(hard_cuts),
        "pose3d": zlib.compress(y3d[:F].astype(np.float32).tobytes()),
        "keypoints2d": zlib.compress(kp2d[:F].astype(np.float32).tobytes()),
        "scores2d": zlib.compress(scores[:F].astype(np.float32).tobytes()),
        "bboxes": zlib.compress(bboxes[:F].astype(np.float32).tobytes()),
        "frame_indices": zlib.compress(np.arange(F, dtype=np.int32).tobytes()),
        "num_pose_frames": F,
    }


def programmatic_grade(row):
    from validate_dataset_quality import programmatic_verdict
    return programmatic_verdict(row)


def vlm_judge(vlm_model, vlm_proc, clip_id, frames, kp2d_coco, sc2d_coco,
              bboxes, instruction, action_class, n_frames=8):
    """Sample 8 frames, overlay skeleton, judge via vLLM server if reachable
    else fall back to local 4-bit Qwen."""
    from validate_dataset_quality import vlm_evaluate_sample
    from scripts.render_skeleton_overlay import draw_skeleton
    from data_prep.keypoints import h36m_coco_format
    import cv2, os, urllib.request
    from PIL import Image

    T = len(frames)
    idxs = np.linspace(0, T - 1, n_frames).astype(int)
    h36m_k, h36m_s, _ = h36m_coco_format(
        kp2d_coco[None, ...], sc2d_coco[None, ...])
    if h36m_k.shape[0] == 0:
        return {"tracking_good": None, "motion_matches": None,
                "explanation": "h36m convert failed", "judge": "none"}
    kp2d_h36m = h36m_k[0]
    sc2d_h36m = h36m_s[0]

    pil_imgs = []
    for i in idxs:
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        overlay = draw_skeleton(bgr, kp2d_h36m[i], sc2d_h36m[i])
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_imgs.append(Image.fromarray(rgb))

    server = os.environ.get("VLM_SERVER_URL")
    if server:
        try:
            urllib.request.urlopen(server.rstrip("/") + "/models", timeout=2)
        except Exception:
            server = None
    if server:
        from scripts.vlm_client import vlm_judge_server
        model_name = os.environ.get("VLM_SERVER_MODEL",
                                    "qwen3-vl-235b-thinking")
        res = vlm_judge_server(pil_imgs, instruction, action_class,
                               server_url=server, model_name=model_name)
        res["judge"] = "server:" + model_name
        return res

    res = vlm_evaluate_sample(
        vlm_model, vlm_proc, pil_imgs, instruction, action_class,
        mode="video")
    res["judge"] = "local:qwen3-vl-32b-4bit"
    return res


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def pick_clips():
    with open(LABELS_PATH) as f:
        data = json.load(f)
    return [e for e in data["labels"]
            if e["label"] in ("needs_retracking", "needs_trimming")]


def load_clip_metadata(clip_ids):
    import pyarrow.parquet as pq
    import pyarrow
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    tables = [pq.read_table(pf, columns=["clip_id", "instruction", "action_class"])
              for pf in files]
    full = pyarrow.concat_tables(tables)
    want = set(clip_ids)
    out = {}
    for cid, inst, act in zip(full.column("clip_id").to_pylist(),
                              full.column("instruction").to_pylist(),
                              full.column("action_class").to_pylist()):
        if cid in want:
            out[cid] = {"instruction": inst, "action_class": act}
    return out


def process_clip(entry, yolo, vlm_model, vlm_proc,
                 vitpose_proc, vitpose_model, model_3d, device, out_dir, meta,
                 skip_vlm_judge=False):
    cid = entry["clip_id"]
    label = entry["label"]
    vp = VIDEO_DIR / f"{cid}.mp4"
    print(f"\n=== {cid} ({label}) ===")

    vm = probe_video_meta_fast(str(vp))
    W, H, fps_src = vm["width"], vm["height"], vm["fps"]
    total_frames = vm["frames"]
    idxs = sample_indices_for_fps(total_frames, fps_src, target_fps=TARGET_FPS)
    frames = read_frames_batch_fast(str(vp), idxs, target_fps=TARGET_FPS)
    T = len(frames)
    print(f"  video: {W}x{H}@{fps_src:.1f}fps, {total_frames}f raw, "
          f"{T}f @{TARGET_FPS}fps")

    # Trim subrange for needs_trimming
    hard_cuts_local = detect_hard_cuts(frames)
    if label == "needs_trimming":
        s, e = choose_trim_range(frames)
        if e - s < T:
            print(f"  trim: keeping [{s}:{e}] ({e - s}/{T})")
            frames = frames[s:e]
            T = len(frames)
            hard_cuts_local = detect_hard_cuts(frames)

    # Multi-frame candidate pool
    sel = np.linspace(0, T - 1, 5).astype(int)
    sel_frames = [frames[i] for i in sel]
    pool, counts = build_candidate_pool(yolo, sel_frames, sel)
    print(f"  yolo per-frame counts (sampled): {counts}; pool={len(pool)}")
    if not pool:
        return {"clip_id": cid, "label": label, "status": "NO_DETECTIONS"}

    # Pick busiest sampled frame (most detections) so oracle actually has
    # choices to disambiguate. Ties broken toward the middle of the clip.
    counts_arr = np.array(counts)
    mid_pref = np.abs(np.arange(len(counts)) - (len(counts) - 1) / 2)
    anchor_sampled_idx = int(np.lexsort((mid_pref, -counts_arr))[0])
    anchor_frame_idx = int(sel[anchor_sampled_idx])
    render_frame = frames[anchor_frame_idx]
    # Draw all pool boxes on the anchor frame (they may originate from other
    # frames; as approximation, we re-detect on the anchor frame only to get
    # correctly-positioned boxes, then match by position to the pool).
    boxes_here, confs_here = yolo_detect(yolo, render_frame)
    if len(boxes_here) == 0:
        # fall back to best-conf sampled frame
        best_count_idx = int(np.argmax(counts))
        anchor_sampled_idx = best_count_idx
        anchor_frame_idx = int(sel[anchor_sampled_idx])
        render_frame = frames[anchor_frame_idx]
        boxes_here, confs_here = yolo_detect(yolo, render_frame)

    if len(boxes_here) == 0:
        return {"clip_id": cid, "label": label, "status": "NO_DETECTIONS_ANCHOR"}

    order = np.argsort(-confs_here)[:MAX_CANDIDATES]
    boxes_here = boxes_here[order]
    confs_here = confs_here[order]
    numbered = [(i + 1, tuple(b.tolist())) for i, b in enumerate(boxes_here)]
    pil_img = draw_numbered(render_frame, numbered)
    render_path = out_dir / f"{cid}_anchor.jpg"
    pil_img.save(render_path, quality=88)

    m = meta[cid]
    server_url = os.environ.get("VLM_SERVER_URL")
    if server_url:
        from scripts.vlm_client import vlm_oracle_pick_server
        res = vlm_oracle_pick_server(
            pil_img, len(boxes_here), m["instruction"], m["action_class"],
            server_url=server_url)
        chosen, reason, raw = res["chosen_index"], res["reason"], res["raw"]
    else:
        chosen, reason, raw = oracle_pick(
            vlm_model, vlm_proc, pil_img, len(boxes_here),
            m["instruction"], m["action_class"])
    print(f"  oracle pick={chosen} / {len(boxes_here)}  reason={reason[:70]}")
    if chosen is None:
        return {"clip_id": cid, "label": label, "status": "ORACLE_FAIL",
                "oracle_raw": raw}

    # Try the oracle's pick first; if the IoU tracker can't sustain a long
    # enough valid run, fall back to the next-highest-confidence candidate
    # (second-choice retry). This rescues clips where the oracle is right
    # semantically but the tracker can't stay locked on that exact YOLO box.
    order_by_conf = np.argsort(-confs_here)  # indices into boxes_here
    # Put the oracle pick first, then others by confidence.
    fallback_order = [chosen - 1] + [i for i in order_by_conf.tolist()
                                      if i != chosen - 1]

    tracked = None
    back_stop = fwd_stop = -1
    used_idx = None
    for attempt, idx in enumerate(fallback_order[:3], 1):
        anchor_box = boxes_here[idx]
        print(f"  tracking attempt {attempt} (candidate {idx + 1})…")
        tr, bs, fs = iou_track(yolo, frames, anchor_frame_idx, anchor_box)
        rl = fs - bs + 1
        print(f"    valid run [{bs}:{fs}] = {rl}/{T}")
        if rl >= MIN_VALID_FRAMES:
            tracked, back_stop, fwd_stop = tr, bs, fs
            used_idx = idx
            if attempt > 1:
                chosen = idx + 1
                reason = f"[fallback#{attempt}] {reason}"
            break
    if tracked is None:
        return {"clip_id": cid, "label": label,
                "status": f"SHORT_RUN:{fs - bs + 1}"}
    anchor_box = boxes_here[used_idx]
    tracked_valid = int(np.isfinite(tracked[:, 0]).sum())
    run_len = fwd_stop - back_stop + 1
    # Trim to longest valid run (always contains anchor with break-on-miss).
    frames, tracked, trim_s, trim_e = trim_to_valid_run(
        frames, tracked, back_stop, fwd_stop)
    T = len(frames)
    anchor_frame_idx -= trim_s
    # Re-detect hard cuts on the trimmed frames so hard_cuts_local matches
    # the kept range (previous value was based on pre-trim frame indices).
    hard_cuts_local = detect_hard_cuts(frames)

    # Further trim within the tracked run to eliminate any hard-cut spans.
    # (Tracker can sometimes bridge a hard cut if the subject happens to be
    # similarly framed before/after. Re-detect and cut.)
    hc_in_run = detect_hard_cuts(frames)
    if hc_in_run:
        edges = [0] + list(sorted(set(hc_in_run))) + [T]
        # Pick the longest cut-free sub-segment that contains anchor_frame_idx.
        best_s, best_e, best_len = 0, T, 0
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i + 1]
            if s <= anchor_frame_idx < e and (e - s) > best_len:
                best_s, best_e, best_len = s, e, e - s
        if best_len >= MIN_VALID_FRAMES and (best_e - best_s) < T:
            print(f"  hard-cut post-trim: [{best_s}:{best_e}] ({best_e - best_s}/{T})")
            frames = frames[best_s:best_e]
            tracked = tracked[best_s:best_e]
            anchor_frame_idx -= best_s
            T = len(frames)
            hard_cuts_local = []  # We just excised them.
        elif best_len < MIN_VALID_FRAMES:
            return {"clip_id": cid, "label": label,
                    "status": f"HARD_CUT_SHORT:{best_len}"}

    # ViTPose
    print("  ViTPose…")
    kp2d, sc2d = run_vitpose(vitpose_proc, vitpose_model, device, frames, tracked)

    # 3D lift
    print("  MotionAGFormer 3D lift…")
    try:
        y3d, seq_k_h36m, seq_s_h36m = lift_3d(kp2d, sc2d, W, H, model_3d, device)
    except Exception as ex:
        return {"clip_id": cid, "label": label,
                "status": f"LIFT_FAIL:{type(ex).__name__}:{str(ex)[:100]}"}

    # Save NPZ
    npz_path = out_dir / f"{cid}.npz"
    np.savez_compressed(
        npz_path,
        pose3d=y3d.astype(np.float32),
        keypoints2d=seq_k_h36m.astype(np.float32),
        scores=seq_s_h36m.astype(np.float32),
        bboxes=tracked.astype(np.float32),
        frame_indices=np.arange(T, dtype=np.int32),
        fps=TARGET_FPS,
        video_width=W,
        video_height=H,
        has_hard_cuts=bool(hard_cuts_local),
        hard_cut_frames=np.array(hard_cuts_local, dtype=np.int32),
    )

    # Build parquet-compatible row for programmatic grade
    row = build_row(cid, TARGET_FPS, W, H, y3d, seq_k_h36m, seq_s_h36m,
                    tracked, m["instruction"], m["action_class"],
                    hard_cuts_local)
    prog_good, issues, _ = programmatic_grade(row)
    print(f"  prog verdict: {prog_good}  issues={issues}")

    # VLM judge on retracked sequence (skipped in batch retrack mode)
    if skip_vlm_judge:
        judge = {"tracking_good": None, "motion_matches": None,
                 "explanation": "skipped (batch re-judge mode)"}
    else:
        print("  VLM judge…")
        try:
            judge = vlm_judge(vlm_model, vlm_proc, cid, frames, kp2d, sc2d,
                              tracked, m["instruction"], m["action_class"])
            print(f"  vlm: tracking={judge.get('tracking_good')} "
                  f"motion={judge.get('motion_matches')}  "
                  f"{str(judge.get('explanation', ''))[:80]}")
        except Exception as ex:
            judge = {"tracking_good": None, "motion_matches": None,
                     "explanation": f"exc:{type(ex).__name__}:{str(ex)[:100]}"}
            print(f"  vlm judge FAILED: {judge['explanation']}")

    return {
        "clip_id": cid,
        "label": label,
        "status": "OK",
        "anchor_frame_idx": anchor_frame_idx,
        "anchor_box": anchor_box.tolist(),
        "oracle_chosen": chosen,
        "oracle_reason": reason,
        "n_candidates": len(boxes_here),
        "tracked_frames": tracked_valid,
        "valid_run": [int(trim_s), int(trim_e)],
        "total_frames": T,
        "programmatic_good": prog_good,
        "issues": issues,
        "vlm_tracking_good": judge.get("tracking_good"),
        "vlm_motion_matches": judge.get("motion_matches"),
        "vlm_explanation": judge.get("explanation", ""),
        "npz": str(npz_path.relative_to(REPO)),
        "render": str(render_path.relative_to(REPO)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/retrack_out")
    ap.add_argument("--out-json", default="tests/retrack_results.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only process these clip_ids")
    ap.add_argument("--clip-list", default=None,
                    help="Path to a text file with one clip_id per line. "
                         "Bypasses manual_labels_bucket1.json; treats all as "
                         "needs_retracking.")
    ap.add_argument("--shard", default="1/1",
                    help="K/N (1-based) — keep clip i iff i %% N == (K-1)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip clips whose NPZ already exists or are present "
                         "in the existing out-json")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--skip-vlm-judge", action="store_true",
                    help="Skip per-clip VLM judge step (do batch re-judge later)")
    args = ap.parse_args()

    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clip_list:
        with open(args.clip_list) as f:
            cids = [line.strip() for line in f if line.strip()]
        clips = [{"clip_id": c, "label": "needs_retracking",
                  "reason": "from --clip-list"} for c in cids]
    else:
        clips = pick_clips()
        if args.only:
            clips = [c for c in clips if c["clip_id"] in set(args.only)]

    # Shard slice
    k, n = [int(x) for x in args.shard.split("/")]
    clips = [c for i, c in enumerate(clips) if i % n == (k - 1)]

    # Resume: load existing results, skip clips already done
    out_json = REPO / args.out_json
    existing_results = []
    existing_cids = set()
    if args.resume and out_json.exists():
        try:
            with open(out_json) as f:
                existing_results = json.load(f).get("results", [])
            existing_cids = {r["clip_id"] for r in existing_results}
        except Exception as ex:
            print(f"Could not read existing {out_json}: {ex}")
    if args.resume:
        # Also treat presence of NPZ as evidence of completion (in case JSON
        # never got flushed)
        for c in clips:
            if (out_dir / f"{c['clip_id']}.npz").exists():
                existing_cids.add(c["clip_id"])
        clips = [c for c in clips if c["clip_id"] not in existing_cids]

    print(f"Shard {k}/{n}: processing {len(clips)} clips "
          f"(resumed: {len(existing_cids)})")

    meta = load_clip_metadata([c["clip_id"] for c in clips])

    print("Loading YOLOv8x…")
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    server_url = os.environ.get("VLM_SERVER_URL")
    vlm_model, vlm_proc = None, None
    if args.skip_vlm_judge:
        # Still need oracle. Prefer server if reachable, else local Qwen.
        if server_url:
            try:
                import urllib.request
                urllib.request.urlopen(server_url.rstrip('/') + "/models",
                                       timeout=3)
                print(f"Using VLM server for ORACLE only at {server_url}")
            except Exception as ex:
                print(f"VLM server unreachable ({ex}); loading local Qwen for oracle")
                server_url = None
        if not server_url:
            print("Loading Qwen3-VL-32B (4-bit) for oracle only…")
            from validate_dataset_quality import load_qwen3_vlm
            vlm_model, vlm_proc = load_qwen3_vlm(device=args.device)
    elif server_url:
        try:
            import urllib.request
            urllib.request.urlopen(server_url.rstrip('/') + "/models",
                                   timeout=3)
            print(f"Using VLM server at {server_url} (skipping local Qwen load)")
        except Exception as ex:
            print(f"VLM server unreachable ({ex}); loading local Qwen 32B (4-bit)")
            server_url = None
    if not server_url and vlm_model is None and not args.skip_vlm_judge:
        print("Loading Qwen3-VL-32B (4-bit)…")
        from validate_dataset_quality import load_qwen3_vlm
        vlm_model, vlm_proc = load_qwen3_vlm(device=args.device)

    print("Loading ViTPose-plus-large…")
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation
    vitpose_proc = AutoImageProcessor.from_pretrained(
        "usyd-community/vitpose-plus-large", trust_remote_code=True)
    vitpose_model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-plus-large", trust_remote_code=True,
        torch_dtype=torch.float32)
    vitpose_model.to(args.device).eval()

    print("Loading MotionAGFormer-B…")
    model_3d = load_motionagformer_from_path(
        "model.MotionAGFormer:MotionAGFormer",
        str(MAGFORMER_CKPT), args.device)

    results = list(existing_results)
    for i, entry in enumerate(clips, 1):
        try:
            r = process_clip(entry, yolo, vlm_model, vlm_proc,
                             vitpose_proc, vitpose_model, model_3d,
                             args.device, out_dir, meta,
                             skip_vlm_judge=args.skip_vlm_judge)
        except Exception as ex:
            import traceback; traceback.print_exc()
            r = {"clip_id": entry["clip_id"], "label": entry["label"],
                 "status": f"EXC:{type(ex).__name__}:{str(ex)[:120]}"}
        results.append(r)
        if i % args.checkpoint_every == 0:
            with open(out_json, "w") as f:
                json.dump({"n": len(results), "results": results}, f)
            print(f"  [shard {k}/{n}] checkpoint at {i}/{len(clips)}")

    with open(out_json, "w") as f:
        json.dump({"n": len(results), "results": results}, f)
    print(f"\nWrote {out_json}")

    ok = [r for r in results if r.get("status") == "OK"]
    good = [r for r in ok if r.get("programmatic_good")]
    print(f"\nSummary: {len(ok)}/{len(results)} retracked; "
          f"{len(good)}/{len(ok)} pass prog")
    for r in results:
        print(f"  {r['clip_id']:<35} {r.get('status','?'):<12} "
              f"prog={r.get('programmatic_good')}  "
              f"issues={r.get('issues', [])}")


if __name__ == "__main__":
    main()
