#!/usr/bin/env python3
"""
Merge the 4 Phase-2 retrack shard JSONs into a single combined file
and emit a clip-id list of successfully-retracked clips for re-judge.

Inputs:
  tests/phase2_results_s1.json ... s4.json
  tests/phase2_npz_s1/ ... s4/

Outputs:
  tests/phase2_results_all.json     # merged, deduped (latest-wins)
  tests/phase2_retracked_ok.txt     # one clip_id per line (status==OK)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    merged = {}
    npz_owners = {}
    for s in (1, 2, 3, 4):
        jpath = REPO / f"tests/phase2_results_s{s}.json"
        npz_dir = REPO / f"tests/phase2_npz_s{s}"
        if jpath.exists():
            with open(jpath) as f:
                d = json.load(f)
            for r in d["results"]:
                merged[r["clip_id"]] = r
                npz_owners[r["clip_id"]] = npz_dir
        else:
            print(f"  WARN: missing {jpath}")

    results = list(merged.values())
    out = {"n": len(results), "results": results}
    out_json = REPO / "tests/phase2_results_all.json"
    with open(out_json, "w") as f:
        json.dump(out, f)

    ok = [r for r in results if r.get("status") == "OK"]
    pg = sum(1 for r in ok if r.get("programmatic_good"))

    # Verify NPZ presence for OK clips
    missing_npz = []
    for r in ok:
        cid = r["clip_id"]
        d = npz_owners.get(cid)
        if d is None or not (d / f"{cid}.npz").exists():
            missing_npz.append(cid)

    ok_with_npz = [r for r in ok if r["clip_id"] not in set(missing_npz)]
    cids_for_judge = sorted(r["clip_id"] for r in ok_with_npz)

    list_path = REPO / "tests/phase2_retracked_ok.txt"
    list_path.write_text("\n".join(cids_for_judge) + "\n")

    print(f"Merged {len(results)} retrack rows from 4 shards")
    print(f"  status=OK:       {len(ok):>6}  ({100*len(ok)/len(results):.1f}%)")
    print(f"  prog_good:       {pg:>6}  ({100*pg/max(len(ok),1):.1f}% of OK)")
    print(f"  OK with NPZ:     {len(ok_with_npz):>6}")
    print(f"  Missing NPZ:     {len(missing_npz):>6}")
    print(f"  Wrote {out_json}")
    print(f"  Wrote {list_path}")


if __name__ == "__main__":
    main()
