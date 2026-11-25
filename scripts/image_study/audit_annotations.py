"""Check boxes and masks against the images. Usage: python -I audit_annotations.py <data/external> <manifest.csv> <out.json>"""
import sys, os, csv, json, collections
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

R, MAN, OUT = sys.argv[1:4]
rows = [r for r in csv.DictReader(open(MAN)) if r['annotation_path']]
res = {}
for ds in ('tn5000', 'tn3k', 'thyroidxl'):
    c = collections.Counter(); frac = []; vals = collections.Counter(); bad = []
    for r in (x for x in rows if x['dataset'] == ds):
        W, H = int(r['width']), int(r['height']); c['n'] += 1
        ap = os.path.join(R, r['annotation_path'])
        if ds == 'tn5000':
            root = ET.parse(ap).getroot(); sz = root.find('size')
            if (int(sz.findtext('width')), int(sz.findtext('height'))) != (W, H): c['xml_size_differs_from_image'] += 1
            b = root.find('object').find('bndbox')
            x0, y0, x1, y1 = (int(float(b.findtext(t))) for t in ('xmin', 'ymin', 'xmax', 'ymax'))
        else:
            m = Image.open(ap)
            if m.size != (W, H): c['mask_size_differs_from_image'] += 1; bad.append(r['sample_id']); continue
            a = np.asarray(m.convert('L'))
            for v in np.unique(a)[:6]: vals[int(v)] += 1
            fg = a > 127
            if not fg.any(): c['empty_mask'] += 1; bad.append(r['sample_id']); continue
            ys, xs = np.where(fg); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if ds == 'thyroidxl':
                o = json.load(open(ap.replace('/masks/', '/labels_orig/').replace('.png', '.json')))
                bx = o['bbox']
                if max(abs(bx[0] - x0), abs(bx[1] - y0), abs(bx[2] - x1), abs(bx[3] - y1)) > 3: c['mask_extent_differs_from_box_by_more_than_3px'] += 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H: c['box_outside_image'] += 1; bad.append(r['sample_id'])
        if x1 - x0 < 8 or y1 - y0 < 8: c['box_smaller_than_8px'] += 1; bad.append(r['sample_id'])
        frac.append((x1 - x0) * (y1 - y0) / (W * H))
    f = np.array(frac)
    res[ds] = dict(counts=dict(c), mask_values_seen=dict(vals), box_area_fraction=dict(min=float(f.min()), median=float(np.median(f)), max=float(f.max())), flagged=bad[:20])
    print(ds, res[ds])
json.dump(res, open(OUT, 'w'), indent=1)
