"""File audit of ThyroidXL. Usage: python -I audit_thyroidxl.py <data/external>"""
import sys, os, hashlib, collections, glob, json
import numpy as np
from PIL import Image

R = sys.argv[1]
D = os.path.join(R, 'thyroidxl')
C = collections.Counter


def dhash(im, n=16):
    a = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


# ---- 1. files, patients, labels
img = {}      # file -> dict(split, pid, label)
for s in ('train', 'test'):
    names = {k: {os.path.splitext(f)[0] for f in os.listdir(os.path.join(D, s, k))}
             for k in ('images', 'masks', 'labels', 'labels_orig')}
    print(s, {k: len(v) for k, v in names.items()}, 'all four sets equal:', len({frozenset(v) for v in names.values()}) == 1)
    A = json.load(open(os.path.join(D, s, f'{s}_annotations.json')))
    cats = {c['id']: c['name'] for c in A['categories']}
    im_by_id = {i['id']: i for i in A['images']}
    per_img = collections.defaultdict(list)
    for a in A['annotations']:
        per_img[a['image_id']].append(a)
    print(' categories', cats, '| annotations per image', dict(C(len(v) for v in per_img.values())),
          '| images without annotation', len(im_by_id) - len(per_img))
    listed = [l.strip() for l in open(os.path.join(D, 'stats', f'{s}_patients.txt')) if l.strip()]
    info = A['info']
    mism = C(); orig = C(); nlines = C()
    for i in A['images']:
        f = i['file_name']; stem = os.path.splitext(f)[0]; pid = stem.split('_')[0]
        lab = per_img[i['id']][0]['category_id']
        lines = open(os.path.join(D, s, 'labels', stem + '.txt')).read().split('\n')
        lines = [l for l in lines if l.strip()]; nlines[len(lines)] += 1
        yolo = {int(l.split()[0]) for l in lines}
        o = json.load(open(os.path.join(D, s, 'labels_orig', stem + '.json')))
        ol = o['label'] if isinstance(o, dict) else tuple(sorted({x['label'] for x in o}))
        orig[(ol, lab)] += 1
        if yolo != {lab}: mism['yolo_vs_coco'] += 1
        if int(pid) != i['patient_id']: mism['pid_field_vs_filename'] += 1
        if f not in info.get(pid, {}).get('images', []): mism['image_not_listed_under_patient'] += 1
        w, h = Image.open(os.path.join(D, s, 'images', f)).size
        if (w, h) != (i['width'], i['height']): mism['size_field'] += 1
        img[f] = dict(split=s, pid=pid, label=lab, nod=stem.split('_')[-1])
    pids = {v['pid'] for v in img.values() if v['split'] == s}
    print(' patients: in filenames', len(pids), '| in list', len(set(listed)), '| in info', len(info),
          '| all equal', pids == set(listed) == set(info))
    print(' label lines per txt', dict(nlines), '| labels_orig label vs category', dict(orig), '| mismatches', dict(mism))
    # patient-level label
    pl = collections.defaultdict(set)
    for v in img.values():
        if v['split'] == s: pl[v['pid']].add(v['label'])
    k = C('both' if len(x) == 2 else cats[next(iter(x))] for x in pl.values())
    print(' patients by image labels', dict(k), '| images by label', dict(C(cats[v['label']] for v in img.values() if v['split'] == s)))
    print(' images per patient: median', int(np.median([sum(1 for v in img.values() if v['pid'] == p) for p in list(pids)[:500]])),
          '(first 500 patients)')
    # clinical fields
    concl = C(); fn = C(); hist = C(); cross = C(); two = 0; tir = C()
    for pid, r in info.items():
        concl[r.get('conclusion')] += 1
        if r.get('nodule_2'): two += 1
        for nk in ('nodule_1', 'nodule_2'):
            n = r.get(nk)
            if not n: continue
            fn[str(n.get('FNAC'))] += 1; hist['present' if n.get('Histopathology') else 'none'] += 1
            tir[str(n.get('TIRADS'))] += 1
            cross[(str(n.get('FNAC')), str(n.get('Conclusion')))] += 1
    print(' patient conclusion codes', dict(concl), '| patients with a second nodule', two)
    print(' nodule FNAC', dict(fn)); print(' nodule histopathology', dict(hist)); print(' nodule TIRADS', dict(tir))
    print(' FNAC x nodule conclusion:'); [print('   ', k, v) for k, v in sorted(cross.items(), key=lambda x: -x[1])]
    # image label vs patient conclusion code
    print(' image label x patient conclusion code', dict(C((cats[v['label']], info[v['pid']].get('conclusion')) for v in img.values() if v['split'] == s)))

tr = {v['pid'] for v in img.values() if v['split'] == 'train'}; te = {v['pid'] for v in img.values() if v['split'] == 'test'}
print('patients in both splits:', len(tr & te))

# ---- 2. duplicates inside ThyroidXL and against the other datasets
sets = {
 'txl_train': sorted(glob.glob(os.path.join(D, 'train/images/*'))),
 'txl_test': sorted(glob.glob(os.path.join(D, 'test/images/*'))),
 'tn3k': sorted(glob.glob(os.path.join(R, 'tn3k/files/*-image/*'))),
 'tn5000': sorted(glob.glob(os.path.join(R, 'tn5000/TN5000_forReview/JPEGImages/*'))),
 'thyus_b1': sorted(glob.glob(os.path.join(R, 'thyus2path/files/batch1_image/dataset/*'))),
 'thyus_b2': sorted(glob.glob(os.path.join(R, 'thyus2path/files/batch2_image/dataset/*'))),
}
rec = []
for s, ps in sets.items():
    sizes = C(); modes = C()
    for p in ps:
        b = open(p, 'rb').read(); im = Image.open(p); sizes[im.size] += 1; modes[im.mode] += 1
        rec.append((s, os.path.basename(p), hashlib.sha256(b).hexdigest(), dhash(im)))
    print(s, 'n', len(ps), 'modes', dict(modes), 'distinct sizes', len(sizes), 'top', sizes.most_common(3))


def describe(i, j):
    """for two ThyroidXL records: same patient? same label? same split?"""
    a, b = img[rec[i][1]], img[rec[j][1]]
    return ('same_patient' if a['pid'] == b['pid'] else 'diff_patient',
            'same_label' if a['label'] == b['label'] else 'diff_label',
            'same_split' if a['split'] == b['split'] else 'cross_split')


by = collections.defaultdict(list)
for i, r in enumerate(rec): by[r[2]].append(i)
ex = C(); exd = C()
for v in by.values():
    if len(v) > 1:
        ex['+'.join(sorted({rec[i][0] for i in v}))] += 1
        t = [i for i in v if rec[i][0].startswith('txl')]
        for a in range(len(t)):
            for b in range(a + 1, len(t)): exd[describe(t[a], t[b])] += 1
print('exact-duplicate groups by dataset combination:', dict(ex))
print('exact-duplicate ThyroidXL pairs:', dict(exd))
M = np.stack([r[3] for r in rec]).astype(np.float32)
istx = np.array([r[0].startswith('txl') for r in rec])
for thr in (0, 6, 12):
    c = C(); cd = C(); exs = collections.defaultdict(list)
    for a in np.where(istx)[0][::400]:
        blk = np.arange(a, min(a + 400, len(rec))); blk = blk[istx[blk]]
        d = M.shape[1] - (M[blk] @ M.T + (1 - M[blk]) @ (1 - M.T))
        ii, jj = np.where(d <= thr)
        for x, j in zip(ii, jj):
            i = blk[x]
            if i == j or rec[i][2] == rec[j][2] or (istx[j] and j < i): continue
            k = '+'.join(sorted({rec[i][0], rec[j][0]})); c[k] += 1
            if istx[j]: cd[describe(i, j)] += 1
            if len(exs[k]) < 2: exs[k].append((rec[i][1], rec[j][1], int(d[x, j])))
    print(f'near-duplicate pairs with a ThyroidXL image (not byte-identical), dHash256 Hamming<={thr}:', dict(c))
    print('   ThyroidXL-internal pairs:', dict(cd))
    if thr == 12: print('   examples', dict(exs))
