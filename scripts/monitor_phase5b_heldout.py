"""DISABLED: archived Phase 5B watchdog; do not use for submissions.

Kaggle CLI accepts a /version suffix but ignores it for session status, logs,
and output. Multiple versions can run concurrently, so this script cannot
safely decide when to retry or advance. Its main entry point fails closed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "phase5b_watchdog_state.json"
STOP = ROOT / "tmp" / "phase5b_watchdog.stop"
INTERVAL = timedelta(minutes=30)
MAX_STARTUP_RETRIES = 2
EXPECTED_MANIFEST_SHA256 = "cce4c8c35cea378a6f2bd040e5468692a04f7e66f324aaddaedba33e63188b01"
EXPECTED_LOCK_SHA256 = "3795454f60770432de3aa290bc98492d5a8a3f84c5028e619ac72b08d39690e9"
OUTPUT_PATTERN = r"results/phase5b_heldout_[0-9]{2}/(results\.jsonl|phase5b_summary\.json)$"


def now() -> datetime:
    return datetime.now(timezone.utc)


def stamp() -> str:
    return now().isoformat(timespec="seconds")


def say(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state: dict) -> None:
    state["updated_utc"] = stamp()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATE)


def run(*argv: str) -> str:
    result = subprocess.run(
        list(argv), cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"{' '.join(argv)} exited {result.returncode}: {output[-2000:]}")
    return output


def slug(batch: int, version: int | None = None) -> str:
    base = f"ajjisan/phase-5b-heldout-{batch:02d}"
    return f"{base}/{version}" if version is not None else base


def status(batch: int, version: int) -> str:
    response = run("kaggle", "kernels", "status", slug(batch, version)).upper()
    for label in ("COMPLETE", "ERROR", "CANCELLED", "FAILED", "RUNNING", "PENDING"):
        if f"KERNELWORKERSTATUS.{label}" in response:
            return label
    raise RuntimeError(f"unrecognized Kaggle status: {response}")


def locked_inputs_ok() -> None:
    manifest = ROOT / "config" / "phase5_prompt_manifest.json"
    lock_path = ROOT / "results" / "phase5b_calibration_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if digest(manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("locked prompt manifest hash changed")
    if digest(lock_path) != EXPECTED_LOCK_SHA256 or lock["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("calibration lock changed")


def audit_batch(batch: int) -> None:
    output = run(sys.executable, "scripts/audit_phase5b_batch.py", str(batch))
    audit_path = ROOT / "results" / f"phase5b_heldout_{batch:02d}" / "heldout_batch_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["verdict"] != "PASS" or audit["complete_cases"] != 9 or audit["segments"] != 36:
        raise RuntimeError(f"batch {batch:02d} audit failed: {output[-1000:]}")
    say(f"batch {batch:02d} audit PASS: 9 cases, 36 segments")


def fetch_and_audit(batch: int, version: int) -> None:
    name = f"phase5b_heldout_{batch:02d}"
    dest = ROOT / "tmp" / "phase5b_downloads" / f"{name}_v{version}"
    output = run(
        "kaggle", "kernels", "output", slug(batch, version),
        "-p", str(dest), "--file-pattern", OUTPUT_PATTERN, "--page-size", "200",
    )
    source = dest / "results" / name
    canonical = ROOT / "results" / name / "results" / name
    for filename in ("results.jsonl", "phase5b_summary.json"):
        path = source / filename
        if not path.is_file():
            raise RuntimeError(f"missing {path}; Kaggle output: {output[-1000:]}")
    summary = json.loads((source / "phase5b_summary.json").read_text(encoding="utf-8"))
    if summary.get("verdict") != "PASS" or summary.get("n_complete") != 9 or summary.get("n_failed_or_incomplete") != 0:
        raise RuntimeError(f"{name} summary is incomplete: {summary.get('verdict')}, {summary.get('n_complete')}/9")
    canonical.mkdir(parents=True, exist_ok=True)
    for filename in ("results.jsonl", "phase5b_summary.json"):
        target = canonical / filename
        source_file = source / filename
        if target.exists() and digest(target) != digest(source_file):
            raise RuntimeError(f"refusing to overwrite different raw artifact: {target}")
        if not target.exists():
            shutil.copy2(source_file, target)
    audit_batch(batch)


def push(batch: int) -> int:
    job = ROOT / "jobs" / f"phase5b_heldout_{batch:02d}"
    response = run(
        "kaggle", "kernels", "push", "-p", str(job),
        "--accelerator", "TeslaT4", "-t", "10800",
    )
    match = re.search(r"Kernel version (\d+) successfully pushed", response)
    if not match:
        raise RuntimeError(f"cannot identify pushed kernel version: {response}")
    version = int(match.group(1))
    say(f"submitted batch {batch:02d} version {version}")
    return version


def finish(state: dict, reason: str, status_name: str = "stopped") -> None:
    state["status"] = status_name
    state["reason"] = reason
    save(state)
    say(f"{status_name}: {reason}")


def initialize() -> dict:
    if STATE.exists():
        existing = json.loads(STATE.read_text(encoding="utf-8"))
        if existing.get("status") == "monitoring":
            raise RuntimeError(f"monitor state already active: {STATE}")
    for batch in (0, 1):
        audit_batch(batch)
    state = {
        "status": "monitoring", "batch": 2, "version": 6,
        "startup_retries": 0,
        "next_check_utc": (now() + INTERVAL).isoformat(timespec="seconds"),
        "note": "Version 6 was submitted after v4/v5 errors; track the exact version.",
    }
    save(state)
    say(f"watching batch 02 version 6; first check at {state['next_check_utc']}")
    return state


def resume() -> dict:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("status") != "monitoring":
        raise RuntimeError(f"watchdog state is not active: {state.get('status')}")
    say(f"resuming batch {state['batch']:02d} version {state['version']}; next check at {state['next_check_utc']}")
    return state


def prevent_idle_sleep(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    continuous = 0x80000000
    system_required = 0x00000001
    flags = continuous | system_required if enabled else continuous
    if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
        raise OSError("SetThreadExecutionState failed")


def monitor(state: dict) -> None:
    while state["status"] == "monitoring":
        if STOP.exists():
            finish(state, f"operator stop file found: {STOP}")
            return
        due = datetime.fromisoformat(state["next_check_utc"])
        remaining = (due - now()).total_seconds()
        if remaining > 0:
            time.sleep(min(60, remaining))
            continue
        locked_inputs_ok()
        batch, version = state["batch"], state["version"]
        current = status(batch, version)
        say(f"batch {batch:02d} version {version}: {current}")
        state["last_remote_status"] = current
        state["last_checked_utc"] = stamp()
        if current == "COMPLETE":
            fetch_and_audit(batch, version)
            if batch == 7:
                output = run(sys.executable, "scripts/audit_phase5b_reference.py")
                final_path = ROOT / "results" / "phase5b_audit" / "phase5b_audit.json"
                final = json.loads(final_path.read_text(encoding="utf-8"))
                if final.get("verdict") != "PASS":
                    raise RuntimeError(f"final Phase 5B audit failed: {output[-1000:]}")
                finish(state, "all eight held-out batches and final Phase 5B audit passed; Phase 5C not started", "complete")
                return
            next_batch = batch + 1
            state.update({
                "batch": next_batch, "version": push(next_batch), "startup_retries": 0,
                "next_check_utc": (now() + INTERVAL).isoformat(timespec="seconds"),
            })
            save(state)
            continue
        if current in ("ERROR", "FAILED", "CANCELLED"):
            logs = run("kaggle", "kernels", "logs", slug(batch, version)).strip()
            if current == "ERROR" and logs == "[]" and state["startup_retries"] < MAX_STARTUP_RETRIES:
                state["startup_retries"] += 1
                state["version"] = push(batch)
                state["next_check_utc"] = (now() + INTERVAL).isoformat(timespec="seconds")
                save(state)
                continue
            finish(state, f"batch {batch:02d} version {version} {current}; logs={logs[-1000:]}")
            return
        state["next_check_utc"] = (now() + INTERVAL).isoformat(timespec="seconds")
        save(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="read-only preflight and exact version 6 status")
    parser.add_argument("--resume", action="store_true", help="resume the active state after an intentional process restart")
    args = parser.parse_args()
    raise RuntimeError(
        "Disabled: Kaggle CLI kernels status/logs/output ignore the /version suffix. "
        "Multiple versions may run concurrently; automatic retry or advance is unsafe."
    )
    locked_inputs_ok()
    if args.check_only:
        print(json.dumps({"batch": 2, "version": 6, "status": status(2, 6), "manifest_and_lock": "unchanged"}))
        return
    state = resume() if args.resume else initialize()
    try:
        prevent_idle_sleep(True)
        monitor(state)
    except Exception as exc:
        finish(state, f"watchdog exception: {type(exc).__name__}: {exc}")
        raise
    finally:
        prevent_idle_sleep(False)


if __name__ == "__main__":
    main()
