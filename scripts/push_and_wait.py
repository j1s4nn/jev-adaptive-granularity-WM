"""
Push and Wait Script
====================

Automates Kaggle job submission and polling.

Usage:
    python scripts/push_and_wait.py --job jobs/phase0 --timeout 7200
    python scripts/push_and_wait.py --job jobs/e1_baseline --parallel 4
"""

import argparse
import subprocess
import time
import json
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, cwd=None):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def push_kernel(job_dir: Path):
    """Push Kaggle kernel."""
    print(f"Pushing kernel from {job_dir}...")

    # Copy src/ into job directory temporarily
    import shutil
    src_dest = job_dir / "src"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    shutil.copytree("src", src_dest)

    returncode, stdout, stderr = run_command(
        "kaggle kernels push --accelerator GPU-T4",
        cwd=job_dir
    )

    # Clean up
    shutil.rmtree(src_dest)

    if returncode != 0:
        print(f"[X] Push failed: {stderr}")
        return None

    print(f"[OK] Kernel pushed")

    # CLI 2.x: extract the real slug from the progress URL (title-derived slug
    # may differ from metadata id)
    for line in stdout.splitlines():
        if "kaggle.com/code/" in line:
            import re
            m = re.search(r"kaggle\.com/code/([\w\-]+/[\w\-]+)", line)
            if m:
                print(f"[OK] Kernel slug: {m.group(1)}")
                return m.group(1)

    return None


def get_kernel_status(kernel_slug: str):
    """Check kernel status."""
    returncode, stdout, stderr = run_command(
        f"kaggle kernels status {kernel_slug}"
    )

    if returncode != 0:
        return None

    return stdout.strip()


def download_output(kernel_slug: str, output_dir: Path):
    """Download kernel output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    returncode, stdout, stderr = run_command(
        f"kaggle kernels output {kernel_slug} -p {output_dir}"
    )

    return returncode == 0


def wait_for_completion(kernel_slug: str, timeout: int, poll_interval: int = 60):
    """Poll kernel until complete or timeout."""
    start_time = time.time()

    print(f"Waiting for {kernel_slug} (timeout={timeout}s)...")

    while (time.time() - start_time) < timeout:
        status = get_kernel_status(kernel_slug)

        if status is None:
            print("  [!] Could not get status")
            time.sleep(poll_interval)
            continue

        print(f"  Status: {status} (elapsed: {int(time.time() - start_time)}s)")

        # Normalize CLI 1.x ("complete") and CLI 2.x ("KernelWorkerStatus.COMPLETE")
        status_lower = status.lower()
        if "complete" in status_lower:
            print("[OK] Kernel complete!")
            return True
        elif "error" in status_lower or "failed" in status_lower or "cancelled" in status_lower:
            print(f"[X] Kernel {status}")
            return False

        time.sleep(poll_interval)

    print(f"[X] Timeout after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Push Kaggle kernel and wait for completion")
    parser.add_argument("--job", type=str, required=True, help="Job directory (e.g., jobs/phase0)")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--download", action="store_true", help="Download output when complete")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel jobs (not implemented yet)")

    args = parser.parse_args()

    job_dir = Path(args.job)
    if not job_dir.exists():
        print(f"[X] Job directory not found: {job_dir}")
        sys.exit(1)

    # Read kernel metadata
    metadata_file = job_dir / "kernel-metadata.json"
    if not metadata_file.exists():
        print(f"[X] kernel-metadata.json not found in {job_dir}")
        sys.exit(1)

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Push kernel (returns the real slug from CLI 2.x output)
    kernel_slug = push_kernel(job_dir)
    if kernel_slug is None:
        sys.exit(1)

    # Wait for completion
    success = wait_for_completion(kernel_slug, args.timeout, args.poll)

    if not success:
        sys.exit(1)

    # Download output
    if args.download:
        output_dir = Path(f"results/{job_dir.name}")
        print(f"Downloading output to {output_dir}...")

        if download_output(kernel_slug, output_dir):
            print(f"[OK] Output downloaded to {output_dir}")
        else:
            print("[X] Download failed")
            sys.exit(1)

    print("\n[OK] Job complete!")


if __name__ == "__main__":
    main()
