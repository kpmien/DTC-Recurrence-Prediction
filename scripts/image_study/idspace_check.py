# Which patient-ID numbering does the batch-1 label file follow? Fit on batch 2 (consistent IDs), score batch 1.
import sys, os, csv, collections, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
R, W = sys.argv[1], sys.argv[2]
torch.set_num_threads(6); torch.manual_seed(0); rng = np.random.default_rng(0)
proc = AutoImageProcessor.from_pretrained(W); model = AutoModel.from_pretrained(W).eval()
def feats(paths):
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert('RGB') for p in paths[i:i+16]]
        with torch.no_grad(): out.append(model(**proc(images=ims, return_tensors='pt')).pooler_output.numpy())
    return np.concatenate(out)
def patient_feats(D, by_pat, k=5):
    X = {}
    for n, (p, fs) in enumerate(sorted(by_pat.items())):
        fs = sorted(fs); sel = [fs[i] for i in sorted(rng.choice(len(fs), min(k, len(fs)), replace=False))]
        X[p] = feats([os.path.join(D, 'dataset', f) for f in sel]).mean(0)
        if n % 50 == 0: print('  ', n, len(by_pat), flush=True)
    return X
# batch 2: trusted mapping
D2 = os.path.join(R, 'batch2_image'); lab2 = {r['patient_name']: int(r['histo_label']) for r in csv.DictReader(open(os.path.join(D2, 'batch2_image_label.csv')))}
bp2 = collections.defaultdict(list)
for f in os.listdir(os.path.join(D2, 'dataset')): bp2[f.split('_')[0]].append(f)
bp2 = {p: v for p, v in bp2.items() if p in lab2}
# batch 1: patients with a unique image-number fingerprint
D1 = os.path.join(R, 'batch1_image'); lab1 = {r['patient_name']: int(r['histo_label']) for r in csv.DictReader(open(os.path.join(D1, 'batch1_image_label.csv')))}
rows = [r['path'] for r in csv.DictReader(open(os.path.join(D1, 'batch1_image.csv')))]
bp1 = collections.defaultdict(list)
for f in os.listdir(os.path.join(D1, 'dataset')): bp1[f.split('_')[0]].append(f)
def fp(names):
    o = collections.defaultdict(list)
    for n in names: o[n.split('_')[0]].append(int(n.split('_')[1].split('.')[0]))
    return {k: tuple(sorted(v)) for k, v in o.items()}
f_csv, f_file = fp(rows), fp([f for v in bp1.values() for f in v])
inv = collections.defaultdict(list)
for k, v in f_csv.items(): inv[v].append(k)
m = {p: inv[v][0] for p, v in f_file.items() if len(inv[v]) == 1}
print('batch2 patients', len(bp2), 'batch1 uniquely matched patients', len(m), flush=True)
X2 = patient_feats(D2, bp2); X1 = patient_feats(D1, {p: bp1[p] for p in m})
p2 = sorted(X2); A2 = np.stack([X2[p] for p in p2]); y2 = np.array([lab2[p] for p in p2])
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
cv = cross_val_predict(clf, A2, y2, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
print('batch2 internal 5-fold patient-level AUROC', round(roc_auc_score(y2, cv), 3))
clf.fit(A2, y2); p1 = sorted(X1); s = clf.predict_proba(np.stack([X1[p] for p in p1]))[:, 1]
yA = np.array([lab1[p] for p in p1]); yB = np.array([lab1[m[p]] for p in p1])
def boot(y):
    a = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if len(set(y[i])) == 2]
    return round(roc_auc_score(y, s), 3), [round(float(np.percentile(a, q)), 3) for q in (2.5, 97.5)]
print('n', len(p1), 'labels differ between the two readings for', int((yA != yB).sum()))
print('batch1 AUROC if label file follows FILENAME ids:', boot(yA))
print('batch1 AUROC if label file follows CSV ids (via fingerprint map):', boot(yB))
d = yA != yB; print('on the', int(d.sum()), 'patients where readings disagree: AUROC filename', round(roc_auc_score(yA[d], s[d]), 3), 'csv', round(roc_auc_score(yB[d], s[d]), 3))
