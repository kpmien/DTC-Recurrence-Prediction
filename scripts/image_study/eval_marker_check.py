"""Compare the caliper detector with the hand labels of the check sample.

Usage: python -I eval_marker_check.py <key.csv> <marker_labels.csv> <marker_flags.csv> <threshold> <out.json>

The scores are the `marker_score` column written by finalize_markers.py (second strongest peak after the
static-furniture filter). The threshold must be the one fixed beforehand on training images (0.7, PROGRESS.md).
Acceptance rule (fixed before the labels existed): in every dataset, sensitivity >= 0.90 and
specificity >= 0.90 on the images the labeller could decide. Sensitivity is not judged where fewer than
10 images carry calipers. "Cannot tell" answers are counted and left out.
"""
import sys, csv, json, collections, math

KEY, LAB, SCO, THR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
key = {r['file']: r for r in csv.DictReader(open(KEY))}
lab = {r['file']: r['calipers'] for r in csv.DictReader(open(LAB))}
score = {r['sample_id']: float(r['marker_score']) for r in csv.DictReader(open(SCO))}


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - s) / d, 3), round((c + s) / d, 3)]


res = {}; ok = True; wrong = []
for ds in sorted({r['dataset'] for r in key.values()}):
    c = collections.Counter()
    for f, r in key.items():
        if r['dataset'] != ds: continue
        y = lab.get(f, '')
        if y not in ('yes', 'no'): c['unsure' if y == 'unsure' else 'missing'] += 1; continue
        pred = score[r['sample_id']] >= THR
        k = ('tp' if pred else 'fn') if y == 'yes' else ('fp' if pred else 'tn'); c[k] += 1
        if k in ('fn', 'fp'): wrong.append((ds, f, k, round(score[r['sample_id']], 3)))
    pos, neg = c['tp'] + c['fn'], c['tn'] + c['fp']
    sens = c['tp'] / pos if pos else None; spec = c['tn'] / neg if neg else None
    passed = (spec is not None and spec >= 0.90) and (pos < 10 or sens >= 0.90) and c['missing'] == 0
    ok &= passed
    res[ds] = dict(counts=dict(c), hand_prevalence=round(pos / (pos + neg), 3) if pos + neg else None,
                   sensitivity=None if sens is None else round(sens, 3), sensitivity_ci95=wilson(c['tp'], pos),
                   specificity=None if spec is None else round(spec, 3), specificity_ci95=wilson(c['tn'], neg), passed=passed)
    print(ds, res[ds])
json.dump(dict(threshold=THR, accepted=bool(ok), datasets=res, disagreements=wrong), open(OUT, 'w'), indent=1)
print('ACCEPTED' if ok else 'NOT ACCEPTED', '| disagreements', len(wrong))
