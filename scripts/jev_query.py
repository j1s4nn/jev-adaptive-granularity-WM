"""E-JEV-1 step 2: query Jev for each segment state.

Reads the API key from the TYPESAFE_API_KEY environment variable (never hardcoded;
the key file lives outside the repo at %USERPROFILE%\\.kilo-automation\\typesafe.env).

Reads  results/jev_signal/states.jsonl
Writes results/jev_signal/jev_answers.jsonl
"""

import json
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES = os.path.join(ROOT, "results", "jev_signal", "states.jsonl")
OUT = os.path.join(ROOT, "results", "jev_signal", "jev_answers.jsonl")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"

QUESTIONS = {
    "coarse_drift": {
        "type": "noul",
        "instructions": "Given this video context, would generating the next few frames with a coarse 3-frame-chunk method likely drift or become temporally inconsistent compared with a frame-wise method?",
        "criteria": {
            "true": "Fast or complex motion, occlusion, or a scene change that a coarse chunk would blur or smear",
            "false": "Simple, slow, or static motion that a coarse chunk would reproduce faithfully",
        },
    },
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it to predict the next few frames of this video accurately?",
        "criteria": ["very easy", "easy", "moderate", "hard", "very hard"],
    },
    "motion": {
        "type": "choice",
        "instructions": "What kind of motion best describes this context?",
        "criteria": {
            "static": "Almost no motion",
            "moderate": "Some smooth motion",
            "fast": "Fast or complex motion",
            "scene_change": "A cut or large scene change",
        },
    },
}


def build_state(row):
    return {
        "video_prompt": row["prompt"],
        "context_caption": row["caption"],
        "segment_index": row["segment_index"],
        "horizon_frames": row["horizon"],
        "recent_motion_magnitude": round(row["motion"], 3),
        "recent_temporal_warp": round(row["warp"], 3),
    }


def main():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set in environment")

    rows = []
    with open(STATES, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    latencies = []
    with open(OUT, "w", encoding="utf-8") as fh:
        for row in rows:
            payload = {"model": "jev-latest", "state": build_state(row), "questions": QUESTIONS}
            started = time.time()
            answer = None
            error = None
            for attempt in range(4):
                try:
                    resp = requests.post(ENDPOINT, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, json=payload, timeout=40)
                    if resp.status_code in (429, 529):
                        time.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    answer = resp.json()
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(1 + attempt)
            latency = time.time() - started
            latencies.append(latency)
            record = {
                "segment_id": row["segment_id"], "prompt_idx": row["prompt_idx"], "stratum": row["stratum"],
                "prompt": row["prompt"], "horizon": row["horizon"], "group": row["group"],
                "coarse_unsafe": row["coarse_unsafe"], "fine_gap": row["fine_gap"],
                "regret_fine": row["regret_fine"], "regret_coarse": row["regret_coarse"],
                "warp": row["warp"], "motion": row["motion"], "complexity": row["complexity"],
                "latency_s": latency, "answer": answer, "error": error,
            }
            results.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if answer:
                a = answer["answers"]
                print(f"{row['segment_id']}: drift={a['coarse_drift']['noul']:.2f} diff={a['difficulty']['score']:.2f} motion={a['motion']['choice']} unsafe={row['coarse_unsafe']} ({latency:.2f}s)")
            else:
                print(f"{row['segment_id']}: ERROR {error}")

    ok = sum(1 for r in results if r["answer"])
    print(f"\nwrote {len(results)} answers ({ok} ok) -> {os.path.relpath(OUT, ROOT)}")
    print(f"latency mean={sum(latencies)/len(latencies):.2f}s max={max(latencies):.2f}s")


if __name__ == "__main__":
    main()
