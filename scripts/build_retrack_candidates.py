#!/usr/bin/env python3
"""
Emit prioritized retrack candidate clip-id lists from the combined
prog+VLM file.

Buckets (in priority order — earliest = highest expected recovery):
  1. prog_pass + VLM_fail   (~33% recovery on 200-bench)
  2. both_fail              (~25% recovery)
  3. VLM_pass + prog_fail   (~12% recovery)

Writes:
  tests/retrack_cands_pp_vf.txt
  tests/retrack_cands_both.txt
  tests/retrack_cands_pf_vp.txt
  tests/retrack_cands_all.txt   (all three concatenated, no duplicates,
                                  ordered by bucket priority)
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default="tests/scale_strat332k_combined.json")
    ap.add_argument("--out-dir", default="tests")
    args = ap.parse_args()

    with open(REPO / args.combined) as f:
        rows = json.load(f)["results"]

    by_bucket = {"pp_vf": [], "both": [], "pf_vp": []}
    for r in rows:
        if not r.get("vlm_judged"):
            continue
        pg = bool(r.get("prog_good"))
        vg = bool(r.get("vlm_good"))
        if pg and not vg:
            by_bucket["pp_vf"].append(r["clip_id"])
        elif vg and not pg:
            by_bucket["pf_vp"].append(r["clip_id"])
        elif (not pg) and (not vg):
            by_bucket["both"].append(r["clip_id"])

    out_dir = REPO / args.out_dir
    for k, lst in by_bucket.items():
        lst.sort()
        path = out_dir / f"retrack_cands_{k}.txt"
        path.write_text("\n".join(lst) + "\n" if lst else "")
        print(f"  {k}: {len(lst):>7}  → {path}")

    # Combined: priority order (best recovery first)
    seen = set()
    ordered = []
    for k in ("pp_vf", "both", "pf_vp"):
        for c in by_bucket[k]:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
    (out_dir / "retrack_cands_all.txt").write_text("\n".join(ordered) + "\n")
    print(f"\n  TOTAL (priority-ordered, dedup): {len(ordered):>7}")
    print(f"  Wrote {out_dir / 'retrack_cands_all.txt'}")


if __name__ == "__main__":
    main()
