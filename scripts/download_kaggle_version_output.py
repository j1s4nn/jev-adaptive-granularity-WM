"""Download the two small output files from one explicit Kaggle kernel version.

The Kaggle CLI ignores /version for `kernels output`. The SDK's
DownloadKernelOutput request has a separate version_number field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
from requests.exceptions import HTTPError


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int, choices=range(8))
    parser.add_argument("version", type=int)
    args = parser.parse_args()

    name = f"phase5b_heldout_{args.batch:02d}"
    slug = f"phase-5b-heldout-{args.batch:02d}"
    output_dir = ROOT / "tmp" / "phase5b_version_outputs" / f"{name}_v{args.version}" / "results" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client().kernels.kernels_api_client
    for filename in ("results.jsonl", "phase5b_summary.json"):
        request = ApiDownloadKernelOutputRequest()
        request.owner_slug = "ajjisan"
        request.kernel_slug = slug
        request.version_number = args.version
        request.file_path = f"results/{name}/{filename}"
        try:
            response = client.download_kernel_output(request)
            response.raise_for_status()
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(
                f"{name} version {args.version}: {filename} unavailable (HTTP {status})"
            ) from None
        if not response.content:
            raise RuntimeError(f"empty output for {name} version {args.version}: {filename}")
        target = output_dir / filename
        if target.exists() and target.read_bytes() != response.content:
            raise RuntimeError(f"refusing to overwrite different file: {target}")
        target.write_bytes(response.content)
        if filename.endswith(".json"):
            json.loads(target.read_text(encoding="utf-8"))
        else:
            for line in target.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
        print(f"downloaded {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
