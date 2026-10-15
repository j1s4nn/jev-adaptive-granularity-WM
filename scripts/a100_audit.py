"""E-JEV-2b audit: A100 (or other non-T4) hardware re-time.

Paste the Colab RESULT_JSON into results/a100_timing/a100_raw.json (the JSON object
after 'RESULT_JSON: '), then run this script.

Reads  results/a100_timing/a100_raw.json
Writes results/a100_timing/a100_audit.json
"""

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "a100_timing", "a100_raw.json")
OUT = os.path.join(ROOT, "results", "a100_timing", "a100_audit.json")

# T4 reference (F1.3 / Phase B): coarse/fine per-forward ratio and the break-even
T4_RATIO = 3.01
BREAK_EVEN = 2.33


def main():
    with open(IN, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    gpu = raw.get("gpu", "unknown")
    ratio = raw.get("ratio")
    fine = raw.get("fine_s_per_fwd")
    coarse = raw.get("coarse_s_per_fwd")

    report = {"phase": "E-JEV-2b-a100-retime", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "gpu": gpu, "fine_s_per_forward": fine, "coarse_s_per_forward": coarse,
              "coarse_over_fine_per_forward": ratio, "break_even_ratio": BREAK_EVEN,
              "t4_reference_ratio": T4_RATIO, "rows": raw.get("rows", [])}

    if ratio is None:
        report["verdict"] = "no ratio in input"
    elif ratio < BREAK_EVEN:
        report["verdict"] = ("TRADE-OFF EXISTS on this GPU: coarse becomes cheaper than fine "
                             f"(ratio {ratio:.2f} < break-even {BREAK_EVEN}). The T4 inversion is hardware-specific.")
        report["coarse_cheaper"] = True
    else:
        report["verdict"] = (f"Inversion persists: ratio {ratio:.2f} >= break-even {BREAK_EVEN}. "
                             "The c3-3 cost inversion is fundamental to the design, not the T4.")
        report["coarse_cheaper"] = False

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("A100 audit:", os.path.relpath(OUT, ROOT))
    print(f"GPU: {gpu}")
    print(f"fine s/fwd={fine:.3f}  coarse s/fwd={coarse:.3f}  ratio={ratio:.2f}  (T4 was {T4_RATIO}, break-even {BREAK_EVEN})")
    print("verdict:", report["verdict"])


if __name__ == "__main__":
    main()
