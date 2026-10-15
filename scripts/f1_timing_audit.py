"""F1 timing audit.

Audits the already-collected Phase 5B paired fine/coarse/reference timing rows
(no GPU needed). Produces a machine-readable audit under results/f1_timing_audit/.

Reads:  results/phase5b_*/**/results.jsonl
Writes: results/f1_timing_audit/f1_timing_audit.json
"""

import glob
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "f1_timing_audit")
MODES = ["reference", "fine", "coarse"]
FIELDS = ["dit_s", "vae_s", "total_s", "peak_vram_gb", "nfe", "checkpoint_load_s", "blocks"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def split_of(batch):
    return "calibration" if "calibration" in batch else "held_out"


def main():
    pattern = os.path.join(ROOT, "results", "phase5b_*", "**", "results.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit("No results.jsonl found under results/phase5b_*")

    rows = []
    inputs = []
    for path in files:
        batch = os.path.relpath(path, os.path.join(ROOT, "results")).split(os.sep)[0]
        inputs.append(
            {
                "batch": batch,
                "path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                "sha256": sha256_file(path),
            }
        )
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") != "complete":
                    continue
                rec["_batch"] = batch
                rec["_split"] = split_of(batch)
                rows.append(rec)

    splits = {"calibration": [], "held_out": []}
    for rec in rows:
        splits[rec["_split"]].append(rec)

    report = {
        "phase": "F1.1-existing-timing-audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metric_note": (
            "Timing rows are from the already-inspected Phase 5B protocol. "
            "This audit only re-summarizes cost; it makes no quality claim and "
            "does not re-score any held-out signal."
        ),
        "inputs": inputs,
        "counts": {k: len(v) for k, v in splits.items()},
        "per_mode": {},
        "fine_minus_coarse": {},
        "fractions": {},
        "horizons": {},
        "segment_observations": {},
    }

    for split, recs in splits.items():
        mode_stats = {}
        for mode in MODES:
            mode_stats[mode] = {
                field: summarize([r.get(mode, {}).get(field) for r in recs])
                for field in FIELDS
            }
        report["per_mode"][split] = mode_stats

        deltas = {}
        for field in ["dit_s", "vae_s", "total_s", "nfe", "peak_vram_gb"]:
            diffs = []
            for r in recs:
                f = r.get("fine", {}).get(field)
                c = r.get("coarse", {}).get(field)
                if isinstance(f, (int, float)) and isinstance(c, (int, float)):
                    diffs.append(f - c)
            deltas[field] = summarize(diffs)
        report["fine_minus_coarse"][split] = deltas

        n = len(recs)
        fine_faster = sum(
            1
            for r in recs
            if isinstance(r.get("fine", {}).get("total_s"), (int, float))
            and isinstance(r.get("coarse", {}).get("total_s"), (int, float))
            and r["fine"]["total_s"] < r["coarse"]["total_s"]
        )
        fine_faster_dit = sum(
            1
            for r in recs
            if isinstance(r.get("fine", {}).get("dit_s"), (int, float))
            and isinstance(r.get("coarse", {}).get("dit_s"), (int, float))
            and r["fine"]["dit_s"] < r["coarse"]["dit_s"]
        )
        report["fractions"][split] = {
            "n": n,
            "fine_faster_total": fine_faster,
            "fine_faster_total_frac": fine_faster / n if n else None,
            "fine_faster_dit": fine_faster_dit,
            "fine_faster_dit_frac": fine_faster_dit / n if n else None,
        }

        horizons = {}
        for r in recs:
            h = r.get("horizon")
            horizons.setdefault(h, []).append(r)
        report["horizons"][split] = {
            str(h): {
                "n": len(rs),
                "fine_total_s_mean": summarize([x.get("fine", {}).get("total_s") for x in rs])["mean"],
                "coarse_total_s_mean": summarize([x.get("coarse", {}).get("total_s") for x in rs])["mean"],
            }
            for h, rs in horizons.items()
        }

        obs = [o for r in recs for o in r.get("observations", [])]
        report["segment_observations"][split] = {
            "n_segments": len(obs),
            "signal_s": summarize([o.get("signal_s") for o in obs]),
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "f1_timing_audit.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("F1.1 timing audit written:", os.path.relpath(out_path, ROOT))
    for split in ["calibration", "held_out"]:
        f = report["fractions"][split]
        tm = report["per_mode"][split]
        print(
            f"  {split}: n={f['n']} | fine_total_mean={tm['fine']['total_s']['mean']:.2f}s "
            f"coarse_total_mean={tm['coarse']['total_s']['mean']:.2f}s | "
            f"fine_faster_total={f['fine_faster_total']}/{f['n']} | "
            f"fine_dit_mean={tm['fine']['dit_s']['mean']:.2f}s "
            f"coarse_dit_mean={tm['coarse']['dit_s']['mean']:.2f}s | "
            f"nfe fine/coarse={tm['fine']['nfe']['mean']:.1f}/{tm['coarse']['nfe']['mean']:.1f}"
        )


if __name__ == "__main__":
    main()
