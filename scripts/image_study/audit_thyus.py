import sys, os, hashlib, collections, csv, pickle
import numpy as np
from PIL import Image
R, OUT = sys.argv[1], sys.argv[2]
def dhash(im, n=16):
    a = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)
rec = []
for b in ('batch1_image', 'batch2_image'):
    D = os.path.join(R, 'thyus2path', 'files', b)
    lab = {r['patient_name']: r['histo_label'] for r in csv.DictReader(open(os.path.join(D, b + '_label.csv')))}
    rows = list(csv.DictReader(open(os.path.join(D, b + '.csv'))))
    files = set(os.listdir(os.path.join(D, 'dataset'))); low = {f.lower(): f for f in files}
    for r in rows: r['path'] = low.get(r['path'].lower(), r['path'])
    print('==', b, 'csv rows', len(rows), 'files', len(files), 'rows without file', sum(r['path'] not in files for r in rows), 'files not in csv', len(files - {r['path'] for r in rows}))
    pats = collections.Counter(r['patient_name'] for r in rows)
    print('patients in image csv', len(pats), 'in label csv', len(lab), 'image patients without label', len(set(pats) - set(lab)), 'label patients without images', len(set(lab) - set(pats)))
    print('label values (patients)', collections.Counter(lab.values()), '| images by label', collections.Counter(lab.get(r['patient_name']) for r in rows))
    n = np.array(list(pats.values())); print('images per patient: min', n.min(), 'median', np.median(n), 'max', n.max())
    print('prefix of filename == patient', sum(r['path'].split('_')[0] == r['patient_name'] for r in rows), '/', len(rows))
    sizes = collections.Counter(); modes = collections.Counter()
    for r in rows:
        if r['path'] not in files: continue
        p = os.path.join(D, 'dataset', r['path']); by = open(p, 'rb').read(); im = Image.open(p)
        sizes[im.size] += 1; modes[im.mode] += 1
        rec.append(dict(ds='thyus2path', sub=b, id=r['path'], group=b + ':' + r['patient_name'], label=lab.get(r['patient_name']), sha=hashlib.sha256(by).hexdigest(), dh=dhash(im), size=im.size))
    print('modes', modes, 'top sizes', sizes.most_common(6), 'n distinct', len(sizes))
pickle.dump(rec, open(OUT, 'wb'))
# duplicates
by = collections.defaultdict(list)
for r in rec: by[r['sha']].append(r)
g = [v for v in by.values() if len(v) > 1]
print('exact-duplicate groups', len(g), 'surplus files', sum(len(v) - 1 for v in g), '| groups across patients', sum(len({r['group'] for r in v}) > 1 for v in g), '| across batches', sum(len({r['sub'] for r in v}) > 1 for v in g), '| conflicting labels', sum(len({r['label'] for r in v}) > 1 for v in g))
M = np.stack([r['dh'] for r in rec]).astype(np.float32)
for thr in (0, 4, 10):
    tot = xp = xb = xl = 0
    for a in range(0, len(rec), 500):
        d = M.shape[1] - (M[a:a+500] @ M.T + (1 - M[a:a+500]) @ (1 - M.T))
        ii, jj = np.where(d <= thr)
        for i, j in zip(ii + a, jj):
            if i < j:
                tot += 1; xp += rec[i]['group'] != rec[j]['group']; xb += rec[i]['sub'] != rec[j]['sub']; xl += rec[i]['label'] != rec[j]['label']
    print(f'dHash256 Hamming<={thr}: pairs {tot}, across patients {xp}, across batches {xb}, with different labels {xl}')
