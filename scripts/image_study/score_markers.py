"""Run the caliper detector over images and write one row per image.

Usage: python -I score_markers.py <data/external> <manifest.csv> <weights.pt> <out.csv> [design|check <key.csv>]

  (no mode)  every row of the manifest
  design     200 training images per dataset plus 200 ThyUS2Path batch 1 images (for fixing the threshold)
  check      the images of the hand-labelled check sample listed in key.csv
The diagnosis label is not read.
"""
import sys, os, csv, random, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers

R, MAN, WTS, OUT = sys.argv[1:5]
mode = sys.argv[5] if len(sys.argv) > 5 else 'all'
torch.set_num_threads(8)
rows = list(csv.DictReader(open(MAN)))
if mode == 'design':
    rng = random.Random(7); todo = []
    tr = [r for r in rows if r['experimental_split'] == 'dev_a' and r['dev_role'] == 'train']
    for d in ('tn5000', 'tn3k', 'thyroidxl'):
        todo += [(r['sample_id'], d, r['data_path']) for r in rng.sample([r for r in tr if r['dataset'] == d], 200)]
    b1 = 'thyus2path/files/batch1_image/dataset'
    todo += [(f'thyus_b1:{f}', 'thyus_b1', f'{b1}/{f}') for f in rng.sample(sorted(os.listdir(os.path.join(R, b1))), 200)]
elif mode == 'check':
    keep = {r['sample_id'] for r in csv.DictReader(open(sys.argv[6]))}
    todo = [(r['sample_id'], r['dataset'], r['data_path']) for r in rows if r['sample_id'] in keep]
else:
    todo = [(r['sample_id'], r['dataset'], r['data_path']) for r in rows]
net = markers.load(WTS); t = time.time()
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['sample_id', 'dataset', 'data_path', 'score', 'second_peak', 'peaks_ge_0.5', 'colour_fraction', 'peaks'])
    for i, (sid, d, p) in enumerate(todo, 1):
        im = markers.read(os.path.join(R, p)); sc, pk = markers.detect(net, im)
        w.writerow([sid, d, p, f'{sc:.4f}', f'{pk[1][0]:.4f}' if len(pk) > 1 else '0', sum(v >= 0.5 for v, _, _ in pk),
                    f'{markers.colour_fraction(im):.4f}', ';'.join(f'{v:.2f}:{x}:{y}' for v, x, y in pk)])
        if i % 2000 == 0: print(i, len(todo), f'{time.time() - t:.0f}s', flush=True)
print('scored', len(todo), f'{time.time() - t:.0f}s')
