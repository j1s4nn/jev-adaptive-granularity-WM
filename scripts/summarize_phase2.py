import csv, json, statistics
from pathlib import Path

base = Path('results/phase2_baseline/results/phase2_baseline/20261005_042507')
meta_path = base / '20261005_042507_meta.json'
data = json.loads(meta_path.read_text(encoding='utf-8-sig'))
rows = data['tests']
fields = ['label','prompt','seed','num_frames_requested','decoded_frame_count','mode','nfe_from_recorded_chunk_steps','dit_s','vae_s','combined_generation_s','peak_vram_gb','temporal_mean_l1','temporal_std_l1','latent_temporal_mean_l1','temporal_stability_proxy','ok']
with (base / 'baseline_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        mode = 'c1-1' if 'c1-1' in r['label'] else 'c3-3'
        w.writerow({**{k:r.get(k) for k in fields if k in r}, 'mode':mode,
                    'nfe_from_recorded_chunk_steps':sum(r.get('steps_per_chunk',[])),
                    'combined_generation_s':round(r.get('dit_s',0)+r.get('vae_s',0),3)})
summary=[]
for mode in ('c1-1','c3-3'):
  for horizon in (21,33,45):
    group=[r for r in rows if mode in r['label'] and r['num_frames_requested']==horizon]
    summary.append({'mode':mode,'horizon_frames':horizon,'n':len(group),
      'dit_s_mean':round(statistics.mean(r['dit_s'] for r in group),3),
      'nfe_mean':round(statistics.mean(sum(r.get('steps_per_chunk',[])) for r in group),2),
      'vae_s_mean':round(statistics.mean(r['vae_s'] for r in group),3),
      'dit_plus_vae_s_mean':round(statistics.mean(r['dit_s']+r['vae_s'] for r in group),3),
      'peak_vram_gb_max':max(r['peak_vram_gb'] for r in group),
      'adjacent_frame_l1_mean':round(statistics.mean(r['temporal_mean_l1'] for r in group),6),
      'temporal_proxy_mean':round(statistics.mean(r['temporal_stability_proxy'] for r in group),6)})
with (base/'baseline_summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
(base/'baseline_summary.json').write_text(json.dumps({'source_meta':meta_path.name,'n_success':data['verdict']['n_passed'],'n_total':data['verdict']['n_total'],'summary':summary,'caveats':['two prompts and one seed only; exploratory','adjacent-frame pixel L1 depends on scene motion and is not a perceptual quality score','DiT and VAE times exclude model loading and prompt encoding; report these separately or as unaccounted overhead','no reference videos; no FVD, PSNR, or LPIPS']},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
