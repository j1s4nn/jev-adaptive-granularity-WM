"""Run the remaining Phase 5B Kaggle jobs one at a time and fetch small outputs."""

from __future__ import annotations

import subprocess
import time
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 60
JOB_TIMEOUT_SECONDS = 3 * 60 * 60
OUTPUT_PATTERN = r"results/phase5b_[^/]+/(results\.jsonl|phase5b_summary\.json)$"


def kaggle(*args: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", *args], cwd=ROOT, text=True,
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"kaggle kernels {' '.join(args)} failed: {result.stderr or result.stdout}")
    return result.stdout.strip()


def wait_for(slug: str) -> str:
    started = time.monotonic()
    previous = None
    while time.monotonic() - started < JOB_TIMEOUT_SECONDS:
        status = kaggle("status", slug)
        if status != previous:
            print(status, flush=True)
            previous = status
        upper = status.upper()
        if "COMPLETE" in upper:
            return "COMPLETE"
        if any(word in upper for word in ("ERROR", "FAILED", "CANCELLED")):
            return "ERROR"
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"{slug} did not finish within three hours")


def download(slug: str, name: str, expected: int) -> None:
    dest = ROOT / "results" / name
    dest.mkdir(parents=True, exist_ok=True)
    output = kaggle(
        "output", slug, "-p", str(dest),
        "--file-pattern", OUTPUT_PATTERN, "--page-size", "200",
    )
    print(output, flush=True)
    base = dest / "results" / name
    for filename in ("results.jsonl", "phase5b_summary.json"):
        if not (base / filename).is_file():
            raise FileNotFoundError(base / filename)
    summary = json.loads((base / "phase5b_summary.json").read_text(encoding="utf-8"))
    if summary.get("verdict") != "PASS" or summary.get("n_complete") != expected:
        raise RuntimeError(f"{name} output incomplete: {summary.get('verdict')}, {summary.get('n_complete')}/{expected}")


def main() -> None:
    calibration = "phase5b_reference_calibration_b"
    calibration_slug = "ajjisan/phase-5b-reference-calibration-b"
    print(f"waiting for {calibration_slug}", flush=True)
    if wait_for(calibration_slug) != "COMPLETE":
        raise RuntimeError(f"{calibration_slug} failed")
    download(calibration_slug, calibration, 6)

    for index in range(8):
        name = f"phase5b_heldout_{index:02d}"
        slug = f"ajjisan/phase-5b-heldout-{index:02d}"
        print(f"submitting {name}", flush=True)
        for attempt in range(1, 4):
            print(kaggle(
                "push", "-p", str(ROOT / "jobs" / name),
                "--accelerator", "TeslaT4", "-t", str(JOB_TIMEOUT_SECONDS),
            ), flush=True)
            if wait_for(slug) == "COMPLETE":
                break
            if attempt == 3:
                raise RuntimeError(f"{slug} failed after {attempt} attempts")
            print(f"retrying {slug} after ERROR", flush=True)
            time.sleep(POLL_SECONDS)
        download(slug, name, 9)
    print("all Phase 5B raw jobs complete", flush=True)


if __name__ == "__main__":
    main()
