#!/usr/bin/env python3
"""
Compose the final dataset verdict for all 325k clips:

  For each clip in the original parquet:
    - if a retrack NPZ exists AND the retrack pipeline produced status=OK
      AND the retrack VLM re-judge has a verdict → use the retrack values
    - else → fall back to the Phase-1 (original-parquet) prog + VLM verdict

Emits tests/final_composite.json with per-clip:
  clip_id, source ("retrack" | "original"),
  prog_good, prog_issues,
  vlm_tracking_good, vlm_motion_matches, vlm_good,
  strict (prog AND vlm_good)

Also reads retracked-clip prog verdict from phase2_results_*.json (the
retrack pipeline already ran prog at retrack time).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    print("Loading Phase 1 combined verdicts...")
    with open(REPO / "tests/scale_strat332k_combined.json") as f:
        phase1 = {r["clip_id"]: r for r in json.load(f)["results"]}

    print("Loading Phase 2 retrack rows...")
    with open(REPO / "tests/phase2_results_all.json") as f:
        retrack = {r["clip_id"]: r for r in json.load(f)["results"]}

    print("Loading Phase 2 re-judge VLM verdicts...")
    rejudge = {}
    rj_path = REPO / "tests/phase2_rejudge.json"
    if rj_path.exists():
        with open(rj_path) as f:
            for r in json.load(f)["results"]:
                rejudge[r["clip_id"]] = r
    else:
        print(f"  (none yet — {rj_path} absent)")

    composite = []
    src_counts = {"retrack": 0, "original": 0}
    for cid, p1 in phase1.items():
        rt = retrack.get(cid)
        rj = rejudge.get(cid)
        if (rt is not None
                and rt.get("status") == "OK"
                and rj is not None
                and rj.get("tracking_good") is not None):
            tg = bool(rj.get("tracking_good"))
            mm = bool(rj.get("motion_matches"))
            row = {
                "clip_id": cid,
                "source": "retrack",
                "prog_good": bool(rt.get("programmatic_good")),
                "prog_issues": rt.get("issues", []),
                "vlm_tracking_good": tg,
                "vlm_motion_matches": mm,
                "vlm_good": tg and mm,
                "strict": bool(rt.get("programmatic_good")) and tg and mm,
            }
            src_counts["retrack"] += 1
        else:
            row = {
                "clip_id": cid,
                "source": "original",
                "prog_good": p1.get("prog_good"),
                "prog_issues": p1.get("prog_issues", []),
                "vlm_tracking_good": p1.get("vlm_tracking_good"),
                "vlm_motion_matches": p1.get("vlm_motion_matches"),
                "vlm_good": p1.get("vlm_good"),
                "strict": p1.get("strict"),
            }
            src_counts["original"] += 1
        composite.append(row)

    out = REPO / "tests/final_composite.json"
    with open(out, "w") as f:
        json.dump({"n": len(composite), "results": composite}, f)

    strict = sum(1 for r in composite if r["strict"])
    vg = sum(1 for r in composite if r["vlm_good"])
    pg = sum(1 for r in composite if r["prog_good"])
    print()
    print(f"Final composite: {len(composite):,} clips")
    print(f"  source=retrack:  {src_counts['retrack']:,}")
    print(f"  source=original: {src_counts['original']:,}")
    print(f"  prog_good:       {pg:,} ({100*pg/len(composite):.1f}%)")
    print(f"  vlm_good:        {vg:,} ({100*vg/len(composite):.1f}%)")
    print(f"  STRICT:          {strict:,} ({100*strict/len(composite):.1f}%)")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
