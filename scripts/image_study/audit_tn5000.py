import sys, os, hashlib, collections, glob
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
D = sys.argv[1]
splits = {}
for f in glob.glob(os.path.join(D, 'ImageSets', '**', '*.txt'), recursive=True):
    splits[os.path.splitext(os.path.basename(f))[0]] = [l.strip() for l in open(f) if l.strip()]
print({k: len(v) for k, v in splits.items()})
names = sorted(splits)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print('overlap', names[i], names[j], len(set(splits[names[i]]) & set(splits[names[j]])))
rows = {}
tags = collections.Counter()
for x in sorted(glob.glob(os.path.join(D, 'Annotations', '*.xml'))):
    r = ET.parse(x).getroot()
    for e in r.iter(): tags[e.tag] += 1
    objs = [(o.findtext('name'), [int(float(o.find('bndbox').findtext(t))) for t in ('xmin', 'ymin', 'xmax', 'ymax')]) for o in r.findall('object')]
    sid = os.path.splitext(os.path.basename(x))[0]
    rows[sid] = dict(objs=objs, w=int(r.find('size').findtext('width')), h=int(r.find('size').findtext('height')),
                     fn=r.findtext('filename'), folder=r.findtext('folder'), src=ET.tostring(r.find('source'), encoding='unicode') if r.find('source') is not None else None)
print('xml tags', dict(tags))
print('n objects per image', collections.Counter(len(v['objs']) for v in rows.values()))
print('label names', collections.Counter(o[0] for v in rows.values() for o in v['objs']))
print('images with mixed labels', sum(len({o[0] for o in v['objs']}) > 1 for v in rows.values()))
print('distinct folder/source values', collections.Counter(v['folder'] for v in rows.values()).most_common(5), collections.Counter(v['src'] for v in rows.values()).most_common(3))
print('filename != id', sum(os.path.splitext(v['fn'] or '')[0] != k for k, v in rows.items()))
lab = {k: (max(int(o[0]) for o in v['objs']) if v['objs'] else None) for k, v in rows.items()}
for s in names:
    c = collections.Counter(lab.get(i) for i in splits[s]); print('class counts', s, dict(c))
print('all', collections.Counter(lab.values()))
# hashes
def dhash(im, n=8):
    a = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())
sha = {}; dh = {}; sizes = {}; modes = collections.Counter(); exif = 0
for k in rows:
    p = os.path.join(D, 'JPEGImages', k + '.jpg')
    b = open(p, 'rb').read(); sha[k] = hashlib.sha256(b).hexdigest()
    im = Image.open(p); sizes[k] = im.size; modes[im.mode] += 1
    if im.getexif(): exif += 1
    dh[k] = dhash(im, 16)
print('modes', modes, 'with exif', exif)
print('exact-duplicate files', len(sha) - len(set(sha.values())))
print('size mismatch xml vs image', sum(sizes[k] != (rows[k]['w'], rows[k]['h']) for k in rows))
print('top sizes', collections.Counter(sizes.values()).most_common(8), 'n distinct', len(set(sizes.values())))
ids = sorted(rows); M = np.stack([np.unpackbits(dh[k]) for k in ids]).astype(np.uint8)
where = {i: s for s in names for i in splits[s] if s in ('train', 'val', 'test')}
G = M.astype(np.float32)
for thr in (0, 4, 10, 20):
    tot = cross = 0; ex = []
    for a in range(0, len(ids), 500):
        dist = G.shape[1] - (G[a:a+500] @ G.T + (1 - G[a:a+500]) @ (1 - G.T))
        ii, jj = np.where(dist <= thr)
        for i, j in zip(ii + a, jj):
            if i < j:
                tot += 1
                if where.get(ids[i]) != where.get(ids[j]): cross += 1
                if len(ex) < 6: ex.append((ids[i], ids[j], int(dist[i - a, j])))
    print(f'dHash256 pairs with Hamming<={thr}: total {tot}, crossing official splits {cross}', ex)
# consecutive-ID similarity: is numbering random w.r.t. content?
d_adj = [(M[i] != M[i+1]).sum() for i in range(len(ids) - 1)]
rng = np.random.default_rng(0); perm = rng.permutation(len(ids))
d_rnd = [(M[perm[i]] != M[perm[i+1]]).sum() for i in range(len(ids) - 1)]
print('median dHash distance adjacent IDs', np.median(d_adj), 'random pairs', np.median(d_rnd))
lab_seq = np.array([lab[k] for k in ids]); print('label equal between adjacent IDs', (lab_seq[1:] == lab_seq[:-1]).mean(), 'expected if random', (lab_seq.mean()**2 + (1-lab_seq.mean())**2))
