import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FILES = {
    'W042': 'W042_2025_08_2025-08-09.jsonl',
    'W065': 'W065_2025_08_2025-08-17.jsonl',
    'W06E': 'W06E_2025_12_2025-12-02.jsonl',
}
WEATHER      = {'W042': 'Sunny', 'W065': 'Rainy', 'W06E': 'Snowy'}
MODELS       = ['YOLOv5n', 'YOLOv8n', 'YOLOv10n']
MODEL_COLORS = {'YOLOv5n': '#2196F3', 'YOLOv8n': '#4CAF50', 'YOLOv10n': '#FF9800'}

def load(path, node):
    rows = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get('name') != 'object.detections.all':
            continue
        try:
            val = json.loads(rec['value'])
        except (json.JSONDecodeError, KeyError):
            continue
        img_ts = val.get('image_timestamp_ns')
        for model, md in val.get('models_results', {}).items():
            rows.append({
                'timestamp':          pd.Timestamp(rec['timestamp'], tz='UTC'),
                'model':              model,
                'image_timestamp_ns': img_ts,
                'total_objects':      md.get('total_objects', 0),
                'inference_s':        md.get('inference_time_seconds', np.nan),
                'detections':         md.get('detections', []),
            })
    return pd.DataFrame(rows)

data = {node: load(path, node) for node, path in FILES.items()}

fig, axes = plt.subplots(3, 2, figsize=(16, 13))

for row_idx, node in enumerate(data):
    df      = data[node]
    weather = WEATHER[node]

    pivot = df.pivot_table(
        index='image_timestamp_ns',
        columns='model',
        values='total_objects',
        aggfunc='mean'
    ).dropna(subset=MODELS)

    pivot['disagreement'] = pivot[MODELS].std(axis=1)
    pivot['ts'] = pd.to_datetime(pivot.index, unit='ns', utc=True)
    pivot = pivot.set_index('ts').sort_index()
    thresh = pivot['disagreement'].quantile(0.90)

    print(f"{node} ({weather}):  mean disagreement = {pivot.disagreement.mean():.3f}"
          f"  |  90th pct threshold = {thresh:.3f}"
          f"  |  high-disagreement frames = {(pivot.disagreement > thresh).sum()}")

    ax_left = axes[row_idx, 0]
    for model in MODELS:
        resampled = pivot[model].resample('15min').mean()
        ax_left.plot(resampled.index, resampled.values,
                     color=MODEL_COLORS[model], linewidth=1.4,
                     label=model, alpha=0.85)
    ax_left.set_title(f'{node} - {weather} · Per-Model Counts', fontweight='bold')
    ax_left.set_ylabel('Objects')
    ax_left.legend(fontsize=9)

    ax_right = axes[row_idx, 1]
    rs_disag = pivot['disagreement'].resample('15min').mean()
    ax_right.fill_between(rs_disag.index, rs_disag.values, alpha=0.35, color='#E53935')
    ax_right.plot(rs_disag.index, rs_disag.values, color='#B71C1C', linewidth=1)
    ax_right.fill_between(rs_disag.index, rs_disag.values,
                           where=(rs_disag.values > thresh),
                           alpha=0.6, color='#FF5252',
                           label=f'High disagreement (>p90 = {thresh:.2f})')
    ax_right.axhline(thresh, color='black', linestyle='--', linewidth=1.2)
    ax_right.set_title(f'{node} - {weather} · Model Disagreement (std)', fontweight='bold')
    ax_right.set_ylabel('Std across models')
    ax_right.legend(fontsize=9)

fig.suptitle('Model Disagreement Over Time — High std = Uncertain Frame', fontsize=15)
plt.tight_layout()
plt.savefig('05_disagreement.png', bbox_inches='tight', dpi=130)
plt.close()
print("Saved: 05_disagreement.png")
