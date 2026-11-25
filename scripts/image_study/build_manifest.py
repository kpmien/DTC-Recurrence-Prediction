"""Build the manifest and the fixed splits of the image study.

Usage: python -I build_manifest.py <data/external> <out_manifest.csv> <out_splits.json>

One row per physical image file. Label: 0 benign, 1 malignant.
group_id: patient where an identifier is released (ThyroidXL, ThyUS2Path), else the byte-identical
duplicate group (TN5000), else the image (TN3K).

experimental_split
  dev_a      fitting and model selection; dev_role says train or val
  test       held-out part of a dataset, never used for fitting or selection by any model
  shift_test dataset that no model is trained on (ThyUS2Path batch 2)
  excluded   not used; exclusion_reason says why
"""
import sys, os, csv, json, hashlib, collections
import xml.etree.ElementTree as ET
from PIL import Image

R, OUT, OUTS = sys.argv[1:4]
SEED = 20261123
VAL_FRACTION = 0.20
rows = []


def add(dataset, name, path, label, official, source_unit='', ann='', source_system='unknown', extra=None):
    b = open(os.path.join(R, path), 'rb').read()
    w, h = Image.open(os.path.join(R, path)).size
    r = dict(sample_id=f'{dataset}:{name}', dataset=dataset, source_unit_id=source_unit, group_id='',
             condition_id='', source_system=source_system, official_split=official, data_path=path,
             annotation_path=ann, paired_sample_id='', label=label, content_hash=hashlib.sha256(b).hexdigest(),
             width=w, height=h, label_basis='', experimental_split='', dev_role='', exclusion_reason='')
    r.update(extra or {})
    rows.append(r)
    return r


# ---- TN5000: official image-level split, no patient identifiers
T = 'tn5000/TN5000_forReview'
off = {}
for s in ('train', 'val', 'test'):
    for l in open(os.path.join(R, T, 'ImageSets/Main', s + '.txt')):
        if l.strip():
            assert l.strip() not in off
            off[l.strip()] = s
for f in sorted(os.listdir(os.path.join(R, T, 'JPEGImages'))):
    sid = os.path.splitext(f)[0]
    root = ET.parse(os.path.join(R, T, 'Annotations', sid + '.xml')).getroot()
    labs = [o.findtext('name') for o in root.findall('object')]
    assert len(labs) == 1 and labs[0] in ('0', '1')
    add('tn5000', sid, f'{T}/JPEGImages/{f}', int(labs[0]), off[sid], ann=f'{T}/Annotations/{sid}.xml')

# ---- TN3K: official split stated as patient-level by the authors, no identifiers released
for s, d, lf in (('trainval', 'trainval', 'label4trainval.csv'), ('test', 'test', 'label4test.csv')):
    lab = {a: int(b) for a, b in csv.reader(open(os.path.join(R, 'tn3k/labels', lf)))}
    files = sorted(os.listdir(os.path.join(R, f'tn3k/files/{d}-image')))
    assert set(files) == set(lab)
    for f in files:
        m = f'tn3k/files/{d}-mask/{f}'
        assert os.path.exists(os.path.join(R, m))
        add('tn3k', f'{s}/{f}', f'tn3k/files/{d}-image/{f}', lab[f], s, ann=m)

# ---- ThyUS2Path batch 2: patient-level labels, no official split
B = 'thyus2path/files/batch2_image'
lab = {r['patient_name']: int(r['histo_label']) for r in csv.DictReader(open(os.path.join(R, B, 'batch2_image_label.csv')))}
csv_pat = {r['path'].lower(): r['patient_name'] for r in csv.DictReader(open(os.path.join(R, B, 'batch2_image.csv')))}
for f in sorted(os.listdir(os.path.join(R, B, 'dataset'))):
    pid = f.split('_')[0]
    assert csv_pat.get(f.lower()) == pid, f
    r = add('thyus2path_b2', f, f'{B}/dataset/{f}', lab.get(pid, ''), '', source_unit=pid,
            source_system='Esaote MyLab (paper)')
    if pid not in lab:
        r['experimental_split'] = 'excluded'; r['exclusion_reason'] = 'patient has no label'

# ---- ThyroidXL: official patient-level split
BASIS = {1: 'histopathology', 2: 'fnac', 3: 'fnac', 4: 'histopathology'}
for s in ('train', 'test'):
    A = json.load(open(os.path.join(R, f'thyroidxl/{s}/{s}_annotations.json')))
    ann = {a['image_id']: a['category_id'] for a in A['annotations']}
    for i in sorted(A['images'], key=lambda x: x['file_name']):
        f = i['file_name']; pid = f.split('_')[0]
        add('thyroidxl', f, f'thyroidxl/{s}/images/{f}', int(ann[i['id']]), s, source_unit=pid,
            ann=f'thyroidxl/{s}/masks/{f}', extra=dict(label_basis=BASIS[A['info'][pid]['conclusion']]))

# ---- group_id
dup = collections.defaultdict(list)
for r in rows:
    if r['dataset'] == 'tn5000':
        dup[r['content_hash']].append(r)
for r in rows:
    d = r['dataset']
    if d in ('thyroidxl', 'thyus2path_b2'):
        r['group_id'] = f"{d}:p{r['source_unit_id']}"
    elif d == 'tn5000':
        first = min(x['sample_id'] for x in dup[r['content_hash']])
        r['group_id'] = first if len(dup[r['content_hash']]) > 1 else r['sample_id']
        r['group_id'] = 'g:' + r['group_id'] if len(dup[r['content_hash']]) > 1 else r['group_id']
    else:
        r['group_id'] = r['sample_id']

# ---- experimental split
by_group = collections.defaultdict(list)
for r in rows:
    by_group[r['group_id']].append(r)


def rank(gid):
    return hashlib.sha256(f'{SEED}|{gid}'.encode()).hexdigest()


def carve_val(dataset):
    """Move VAL_FRACTION of the dev groups of a dataset to val, per label stratum, by seeded hash rank."""
    strata = collections.defaultdict(set)
    for r in rows:
        if r['dataset'] == dataset and r['experimental_split'] == 'dev_a':
            strata[r['label']].add(r['group_id'])
    val = set()
    for lab_, gs in strata.items():
        gs = sorted(gs, key=rank)
        val |= set(gs[:round(VAL_FRACTION * len(gs))])
    for r in rows:
        if r['dataset'] == dataset and r['experimental_split'] == 'dev_a':
            r['dev_role'] = 'val' if r['group_id'] in val else 'train'


for r in rows:
    d, o = r['dataset'], r['official_split']
    if r['experimental_split'] == 'excluded':
        continue
    if d == 'thyus2path_b2':
        r['experimental_split'] = 'shift_test'
    elif d in ('tn3k', 'thyroidxl'):
        r['experimental_split'] = 'test' if o == 'test' else 'dev_a'
    elif d == 'tn5000':
        offs = {x['official_split'] for x in by_group[r['group_id']]}
        if o == 'test':
            r['experimental_split'] = 'test'            # official test kept exactly as released
        elif 'test' in offs:
            r['experimental_split'] = 'excluded'; r['exclusion_reason'] = 'byte-identical copy of an official test image'
        else:
            r['experimental_split'] = 'dev_a'
            # a duplicate group that spans train and val goes to train as a whole
            r['dev_role'] = 'train' if 'train' in offs else 'val'

# TN5000: a dev image that is near-identical (dHash-256, Hamming <= NEAR) to an official test image is excluded
import numpy as np
NEAR = 6


def dhash(path, n=16):
    a = np.asarray(Image.open(os.path.join(R, path)).convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.float32)


t5 = [r for r in rows if r['dataset'] == 'tn5000' and r['experimental_split'] in ('dev_a', 'test')]
H = np.stack([dhash(r['data_path']) for r in t5])
te = np.array([r['experimental_split'] == 'test' for r in t5])
dist = H.shape[1] - (H[~te] @ H[te].T + (1 - H[~te]) @ (1 - H[te]).T)
for r, hit in zip([r for r in t5 if r['experimental_split'] == 'dev_a'], (dist <= NEAR).any(axis=1)):
    if hit:
        for x in by_group[r['group_id']]:
            x['experimental_split'] = 'excluded'; x['dev_role'] = ''
            x['exclusion_reason'] = 'near-identical to an official test image'
carve_val('tn3k'); carve_val('thyroidxl')

# every group has one label and one partition
for g, v in by_group.items():
    assert len({x['label'] for x in v}) == 1, g
    parts = {(x['experimental_split'], x['dev_role']) for x in v if x['experimental_split'] != 'excluded'}
    assert len(parts) <= 1, (g, parts)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
rows.sort(key=lambda r: r['sample_id'])
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

summary = collections.OrderedDict()
for r in rows:
    k = f"{r['dataset']}|{r['experimental_split']}|{r['dev_role'] or '-'}"
    s = summary.setdefault(k, dict(images=0, groups=set(), benign=0, malignant=0, ids=hashlib.sha256()))
    s['images'] += 1; s['groups'].add(r['group_id'])
    if r['label'] != '': s['malignant' if r['label'] == 1 else 'benign'] += 1
    s['ids'].update((r['sample_id'] + '\n').encode())
out = dict(seed=SEED, val_fraction=VAL_FRACTION, tn5000_near_duplicate_hamming=NEAR, manifest=os.path.basename(OUT),
           manifest_sha256=hashlib.sha256(open(OUT, 'rb').read()).hexdigest(), rows=len(rows),
           rule='groups ranked by sha256(seed|group_id) inside each label stratum; lowest val_fraction go to val',
           partitions={k: dict(images=v['images'], groups=len(v['groups']), benign_images=v['benign'],
                               malignant_images=v['malignant'], sample_ids_sha256=v['ids'].hexdigest())
                       for k, v in sorted(summary.items())})
json.dump(out, open(OUTS, 'w'), indent=1)
for k, v in out['partitions'].items():
    print(f"{k:38s} images {v['images']:6d} groups {v['groups']:6d} benign {v['benign_images']:6d} malignant {v['malignant_images']:6d}")
print('rows', len(rows), 'manifest sha256', out['manifest_sha256'])
