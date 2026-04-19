#!/usr/bin/env python3
"""
Thin VLM judgment client that talks to a vLLM OpenAI-compatible server
(default http://localhost:8000/v1). Replaces the local-4bit-32B judge in
scripts/retrack_clips.py when VLM_SERVER_URL is set.

Interface mirrors validate_dataset_quality.vlm_evaluate_sample — returns
  {"tracking_good": bool|None, "motion_matches": bool|None,
   "explanation": str}
"""
import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from PIL import Image


def _img_to_data_url(img: Image.Image, max_side: int = 640) -> str:
    img = img.convert("RGB")
    # Downscale to keep multi-image prompts under vLLM's context window.
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b}"


def vlm_oracle_pick_server(pil_image: Image.Image, n_boxes: int,
                           instruction: str, action_class: str,
                           server_url: str | None = None,
                           model_name: str = "qwen3-vl-235b-thinking") -> dict:
    """Server-based oracle. Returns {chosen_index, reason, raw}."""
    import os
    import re
    from openai import OpenAI
    base = server_url or os.environ.get("VLM_SERVER_URL",
                                        "http://localhost:8000/v1")
    client = OpenAI(base_url=base, api_key="EMPTY")
    prompt = (
        f"One video frame with {n_boxes} numbered bounding boxes, each outlining a person.\n"
        f"Action label: {action_class}\n"
        f"Instruction: \"{instruction[:250]}\"\n\n"
        f"TASK: Pick the SINGLE numbered person who is the main subject performing this action "
        f"or who is most likely to perform it in this clip. If multiple people appear similar, "
        f"prefer the largest / most centered / most engaged figure. You MUST return one number "
        f"from 1 to {n_boxes}.\n\n"
        f"Reply with ONLY this JSON: "
        f'{{"chosen_index": <int 1..{n_boxes}>, "reason": "<under 20 words>"}}'
    )
    content = [{"type": "image_url",
                "image_url": {"url": _img_to_data_url(pil_image)}},
               {"type": "text", "text": prompt}]
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.0,
    )
    out = resp.choices[0].message.content.strip()
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
    if chosen is not None and not (1 <= chosen <= n_boxes):
        chosen = None
    return {"chosen_index": chosen, "reason": reason, "raw": out[:240]}


def vlm_judge_server(images: List[Image.Image], instruction: str,
                     action_class: str, server_url: str | None = None,
                     model_name: str = "qwen3-vl-235b-thinking") -> dict:
    from openai import OpenAI
    base = server_url or os.environ.get("VLM_SERVER_URL",
                                        "http://localhost:8000/v1")
    client = OpenAI(base_url=base, api_key="EMPTY")

    prompt = (
        f"{len(images)} video frames (chronological, left→right) with a colored skeleton "
        f"overlay marking the tracked subject (red=right side, blue=left side, green=spine).\n"
        f"Action label: {action_class}\n"
        f"Instruction: \"{instruction[:300]}\"\n\n"
        f"Evaluate strictly. Default to FALSE when uncertain.\n\n"
        f"(1) tracking_consistent — return FALSE if ANY of:\n"
        f"    - skeleton appears on a different person in any frame\n"
        f"    - scene change / hard cut between frames\n"
        f"    - tracked person is off-screen, barely visible, or tiny for most of the clip\n"
        f"    - skeleton floats off the person, or jumps between people\n"
        f"    Return TRUE only if the same clearly-visible person is cleanly tracked throughout.\n\n"
        f"(2) motion_matches — TRUE if the tracked person is broadly performing "
        f"\"{action_class}\" or a plausible subset of the instruction. FALSE if "
        f"mostly still, unrelated, or you can't tell what they're doing.\n\n"
        f"Reply with ONLY this JSON: "
        f'{{"tracking_consistent": true/false, "motion_matches": true/false, '
        f'"explanation": "<under 15 words>"}}'
    )
    content = [{"type": "image_url",
                "image_url": {"url": _img_to_data_url(im)}} for im in images]
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,  # Thinking model needs headroom for reasoning + JSON
        temperature=0.0,
    )
    out = resp.choices[0].message.content.strip()
    try:
        if "{" in out and "}" in out:
            j = json.loads(out[out.index("{"):out.rindex("}") + 1])
            return {
                "tracking_good": bool(j.get("tracking_consistent", False)),
                "motion_matches": bool(j.get("motion_matches", False)),
                "explanation": str(j.get("explanation", ""))[:200],
            }
    except (json.JSONDecodeError, ValueError):
        pass

    tc = re.search(r'"tracking_consistent"\s*:\s*(true|false)', out, re.I)
    mm = re.search(r'"motion_matches"\s*:\s*(true|false)', out, re.I)
    if tc and mm:
        exp = re.search(r'"explanation"\s*:\s*"([^"]{0,200})', out)
        return {
            "tracking_good": tc.group(1).lower() == "true",
            "motion_matches": mm.group(1).lower() == "true",
            "explanation": (exp.group(1) if exp else "") + " [repaired]",
        }
    return {"tracking_good": None, "motion_matches": None,
            "explanation": out[:200]}


def vlm_judge_server_batch(items: list, concurrency: int = 8,
                            server_url: str | None = None,
                            model_name: str = "qwen3-vl-235b-thinking",
                            on_done=None) -> list:
    """Submit multiple judge requests concurrently to the vLLM server.

    items: list of dicts with keys {id, images, instruction, action_class}.
    Returns: list of dicts in ORIGINAL order, each = vlm_judge_server()
             output + {"id": items[i]["id"]}.

    on_done(i, result): optional callback invoked as each item finishes
                        (useful for streaming checkpoints).
    """
    results = [None] * len(items)

    def _one(i, it):
        try:
            r = vlm_judge_server(it["images"], it["instruction"],
                                 it["action_class"],
                                 server_url=server_url,
                                 model_name=model_name)
        except Exception as ex:
            r = {"tracking_good": None, "motion_matches": None,
                 "explanation": f"ERR {type(ex).__name__}: {str(ex)[:160]}"}
        r["id"] = it["id"]
        return i, r

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one, i, it) for i, it in enumerate(items)]
        for f in as_completed(futs):
            i, r = f.result()
            results[i] = r
            if on_done is not None:
                on_done(i, r)
    return results
