"""Leakage audit of the manifest. Exit code 1 if any check fails.

Usage: python -I audit_leakage.py <data/external> <manifest.csv> <splits.json> <out_report.json>

Recomputes hashes from the files; does not trust the manifest columns it checks.
"""
import sys, os, csv, json, hashlib, collections
import numpy as np
from PIL import Image

R, MAN, SPL, OUT = sys.argv[1:5]
rows = list(csv.DictReader(open(MAN)))
checks = collections.OrderedDict()


def part(r):
    return (r['experimental_split'], r['dev_role'])


def dhash(im, n=16):
    a = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


used = [r for r in rows if r['experimental_split'] != 'excluded']
checks['manifest_matches_splits_json'] = hashlib.sha256(open(MAN, 'rb').read()).hexdigest() == json.load(open(SPL))['manifest_sha256']
checks['sample_id_unique'] = len({r['sample_id'] for r in rows}) == len(rows)
checks['every_used_row_has_label_and_partition'] = all(r['label'] in ('0', '1') and r['experimental_split'] in ('dev_a', 'test', 'shift_test') for r in used)
checks['dev_rows_have_role'] = all((r['dev_role'] in ('train', 'val')) == (r['experimental_split'] == 'dev_a') for r in used)

g = collections.defaultdict(set); gl = collections.defaultdict(set)
for r in used:
    g[r['group_id']].add(part(r)); gl[r['group_id']].add(r['label'])
bad = [k for k, v in g.items() if len(v) > 1]
checks['no_group_in_two_partitions'] = not bad
checks['one_label_per_group'] = all(len(v) == 1 for v in gl.values())

# patient identifiers, read again from the filenames
pp = collections.defaultdict(set)
for r in used:
    if r['dataset'] in ('thyroidxl', 'thyus2path_b2'):
        pid = os.path.basename(r['data_path']).split('_')[0]
        pp[(r['dataset'], pid)].add(part(r))
        if r['group_id'] != f"{r['dataset']}:p{pid}": pp[('group_id_mismatch', r['sample_id'])] = {1, 2}
checks['no_patient_in_two_partitions'] = all(len(v) == 1 for v in pp.values())

# content, recomputed from the files
sha = {}; dh = []
for r in rows:
    p = os.path.join(R, r['data_path']); b = open(p, 'rb').read()
    sha[r['sample_id']] = hashlib.sha256(b).hexdigest()
    dh.append(dhash(Image.open(p)))
checks['content_hash_matches_files'] = all(sha[r['sample_id']] == r['content_hash'] for r in rows)
h = collections.defaultdict(set); hd = collections.defaultdict(set)
for r in used:
    h[sha[r['sample_id']]].add(part(r)); hd[sha[r['sample_id']]].add(r['dataset'])
cross = [k for k, v in h.items() if len(v) > 1]
checks['no_identical_file_in_two_partitions'] = not cross
checks['no_identical_file_in_two_datasets'] = all(len(v) == 1 for v in hd.values())

# near-duplicates (dHash-256, Hamming <= 6) that cross a partition boundary: reported, not a failure by itself
idx = [i for i, r in enumerate(rows) if r['experimental_split'] != 'excluded']
M = np.stack([dh[i] for i in idx]).astype(np.float32)
near = collections.Counter(); ex = []
for a in range(0, len(idx), 500):
    d = M.shape[1] - (M[a:a + 500] @ M.T + (1 - M[a:a + 500]) @ (1 - M.T))
    ii, jj = np.where(d <= 6)
    for i, j in zip(ii + a, jj):
        if i < j:
            x, y = rows[idx[i]], rows[idx[j]]
            if x['group_id'] != y['group_id'] and (part(x) != part(y) or x['dataset'] != y['dataset']):
                k = f"{x['dataset']}:{'/'.join(part(x))} ~ {y['dataset']}:{'/'.join(part(y))}"
                near[k] += 1
                if len(ex) < 10: ex.append((x['sample_id'], y['sample_id'], int(d[i - a, j]), x['label'], y['label']))

report = dict(manifest=os.path.basename(MAN), rows=len(rows), used=len(used), checks=checks,
              passed=all(checks.values()),
              groups_in_two_partitions=bad[:10], identical_files_in_two_partitions=len(cross),
              near_duplicate_pairs_across_partitions_hamming_le_6=dict(near), near_duplicate_examples=ex,
              excluded=dict(collections.Counter(r['exclusion_reason'] for r in rows if r['experimental_split'] == 'excluded')))
json.dump(report, open(OUT, 'w'), indent=1)
for k, v in checks.items(): print('PASS' if v else 'FAIL', k)
print('near-duplicate pairs across partitions (Hamming <= 6):', dict(near)); print('examples', ex)
print('excluded', report['excluded'])
print('RESULT', 'PASSED' if report['passed'] else 'FAILED')
sys.exit(0 if report['passed'] else 1)
