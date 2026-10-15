"""F1.4b CLIP-based independent quality endpoint.

VBench-style no-reference metrics computed on the F1.3 videos, independent of
the internal c1-1 reference:
  subject_consistency : mean cosine similarity of CLIP image embeddings of
                        consecutive frames (temporal subject stability; higher = better)
  text_alignment      : mean cosine similarity between the CLIP text embedding of
                        the prompt and each frame's image embedding (prompt fidelity)
  text_alignment_std  : std of per-frame text alignment (stability of prompt fidelity)

Model: openai/clip-vit-base-patch32 (CLIP ViT-B/32), matching the VBench family.

Reads:  results/f1_mode_cost/videos/*.mp4
Writes: results/f1_mode_cost/f1_quality_clip.json
"""

import glob
import json
import os
import re
import statistics
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "results", "f1_mode_cost", "videos")
OUT_JSON = os.path.join(ROOT, "results", "f1_mode_cost", "f1_quality_clip.json")

MODES = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2", "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
NAME_RE = re.compile(r"^(?P<case>.+)_(?P<mode>ref_c1_1_s4|fine_c1_1_s1|fine_c1_1_s2|coarse_c3_3_s4|coarse_c3_3_s2|coarse_c3_3_s1)$")

PROMPTS = {
    "dev_object_seed17": "a banana",
    "dev_motion_seed17": "a bear catching a salmon in its powerful jaws",
    "dev_human_seed17": "A person is air drumming",
}


def _feat(out):
    """transformers 5.x CLIP get_*_features returns a BaseModelOutputWithPooling."""
    if torch.is_tensor(out):
        return out
    return out.pooler_output



def read_frames(path, stride=2):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    per_video = {}
    with torch.no_grad():
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            m = NAME_RE.match(stem)
            if not m:
                continue
            case = m.group("case")
            mode = m.group("mode")
            prompt = PROMPTS.get(re.sub(r"_h\d+$", "", case))
            frames = read_frames(path)
            if not frames:
                continue
            images = [Image.fromarray(f) for f in frames]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            img_emb = _feat(model.get_image_features(**inputs))
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            emb = img_emb.cpu().numpy()
            cons = float(np.mean(np.sum(emb[:-1] * emb[1:], axis=1))) if len(emb) > 1 else None
            ta_mean, ta_std = None, None
            if prompt:
                txt = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
                txt_emb = _feat(model.get_text_features(**txt))
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                sims = (img_emb @ txt_emb.t()).squeeze(-1).cpu().numpy()
                ta_mean = float(np.mean(sims))
                ta_std = float(np.std(sims))
            per_video[stem] = {
                "case": case, "mode": mode, "prompt": prompt, "n_frames_used": len(emb),
                "subject_consistency": cons, "text_alignment": ta_mean, "text_alignment_std": ta_std,
            }
            print(f"{stem}: cons={cons:.4f} ta={ta_mean if ta_mean is None else round(ta_mean,4)}")

    agg = {}
    for mode in MODES:
        rows = [v for v in per_video.values() if v["mode"] == mode]
        agg[mode] = {"n": len(rows)}
        for key in ["subject_consistency", "text_alignment", "text_alignment_std"]:
            vals = [r[key] for r in rows if r[key] is not None]
            agg[mode][key] = statistics.fmean(vals) if vals else None

    report = {
        "phase": "F1.4b-clip-independent-quality",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "openai/clip-vit-base-patch32",
        "note": "Independent of the internal c1-1 reference.",
        "n_videos": len(per_video),
        "by_mode": agg,
        "per_video": per_video,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\nF1.4b CLIP quality written:", os.path.relpath(OUT_JSON, ROOT))
    print(f"{'mode':<18}{'subject_cons':>14}{'text_align':>12}{'ta_std':>9}")
    for mode in MODES:
        a = agg[mode]
        sc = a["subject_consistency"]
        ta = a["text_alignment"]
        tas = a["text_alignment_std"]
        print(
            f"{mode:<18}"
            f"{(sc if sc is not None else float('nan')):>14.4f}"
            f"{(ta if ta is not None else float('nan')):>12.4f}"
            f"{(tas if tas is not None else float('nan')):>9.4f}"
        )


if __name__ == "__main__":
    main()
