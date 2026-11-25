import sys, os, hashlib, collections, glob, json
import numpy as np
from PIL import Image
R = sys.argv[1]
def dhash(im, n=16):
    a = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)
sets = {
 'tn3k_test': glob.glob(os.path.join(R, 'tn3k/files/test-image/*')),
 'tn3k_trainval': glob.glob(os.path.join(R, 'tn3k/files/trainval-image/*')),
 'tn5000': glob.glob(os.path.join(R, 'tn5000/TN5000_forReview/JPEGImages/*')),
 'thyus_b1': glob.glob(os.path.join(R, 'thyus2path/files/batch1_image/dataset/*')),
 'thyus_b2': glob.glob(os.path.join(R, 'thyus2path/files/batch2_image/dataset/*')),
}
rec = []
for s, ps in sets.items():
    sizes = collections.Counter(); modes = collections.Counter()
    for p in sorted(ps):
        b = open(p, 'rb').read(); im = Image.open(p); sizes[im.size] += 1; modes[im.mode] += 1
        rec.append((s, os.path.basename(p), hashlib.sha256(b).hexdigest(), dhash(im)))
    print(s, 'n', len(ps), 'modes', dict(modes), 'distinct sizes', len(sizes), 'top', sizes.most_common(3))
by = collections.defaultdict(list)
for r in rec: by[r[2]].append(r)
pairs = collections.Counter()
for v in by.values():
    if len(v) > 1:
        ss = sorted({r[0] for r in v}); pairs['+'.join(ss)] += 1
print('exact-duplicate groups by dataset combination:', dict(pairs))
ex = [[(r[0], r[1]) for r in v] for v in by.values() if len(v) > 1 and {r[0] for r in v} >= {'tn3k_test', 'tn3k_trainval'}][:5]; print('tn3k test/trainval exact examples', ex)
M = np.stack([r[3] for r in rec]).astype(np.float32); ds = np.array([r[0] for r in rec])
for thr in (0, 6, 12):
    c = collections.Counter(); exs = collections.defaultdict(list)
    for a in range(0, len(rec), 400):
        d = M.shape[1] - (M[a:a+400] @ M.T + (1 - M[a:a+400]) @ (1 - M.T))
        ii, jj = np.where(d <= thr)
        for i, j in zip(ii + a, jj):
            if i < j and rec[i][2] != rec[j][2]:
                k = '+'.join(sorted({rec[i][0], rec[j][0]})); c[k] += 1
                if len(exs[k]) < 2: exs[k].append((rec[i][1], rec[j][1], int(d[i - a, j])))
    print(f'near-duplicate pairs (not byte-identical), dHash256 Hamming<={thr}:', dict(c))
    if thr == 12: print('examples', {k: v for k, v in exs.items() if '+' in k})
