"""Turn raw detector peaks into one caliper flag per image.

Usage: python -I finalize_markers.py <manifest.csv> <scores_all.csv> <threshold> <out.csv> <out_summary.json>

Scanner furniture (logo, body icon, depth scale) sits at the same place in every frame of one format,
calipers do not. For each frame format (dataset, width, height) with at least MIN_GROUP images, a 16 px
cell in which the detector fires in at least STATIC_SHARE of the images is treated as furniture and its
peaks are dropped. Calipers come in pairs, so the image score is the second strongest remaining peak:
an image is flagged when at least two marks reach the threshold. The diagnosis label is not read.
"""
import sys, csv, json, collections

MAN, SCO, THR, OUT, OUTS = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
CELL, MIN_GROUP, STATIC_SHARE, FIRE = 16, 40, 0.05, 0.5

size = {r['sample_id']: (r['dataset'], r['width'], r['height']) for r in csv.DictReader(open(MAN))}
rows = [r for r in csv.DictReader(open(SCO)) if r['sample_id'] in size]
for r in rows:
    r['pk'] = [(float(v), int(x), int(y)) for v, x, y in (p.split(':') for p in r['peaks'].split(';') if p)]
groups = collections.defaultdict(list)
for r in rows: groups[size[r['sample_id']]].append(r)

static = {}; summary = {}
for g, rs in groups.items():
    cells = set()
    if len(rs) >= MIN_GROUP:
        cnt = collections.Counter()
        for r in rs:
            for c in {(x // CELL, y // CELL) for v, x, y in r['pk'] if v >= FIRE}: cnt[c] += 1
        hot = {c for c, n in cnt.items() if n >= STATIC_SHARE * len(rs)}
        cells = {(cx + dx, cy + dy) for cx, cy in hot for dx in (-1, 0, 1) for dy in (-1, 0, 1)}   # one cell of margin
    static[g] = cells

per_ds = collections.defaultdict(lambda: collections.Counter())
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['sample_id', 'dataset', 'marker_score', 'has_caliper', 'n_calipers', 'strongest_peak', 'raw_score', 'peaks_dropped_as_static', 'colour_fraction'])
    for r in rows:
        g = size[r['sample_id']]
        keep = [p for p in r['pk'] if (p[1] // CELL, p[2] // CELL) not in static[g]]
        vals = sorted((p[0] for p in keep), reverse=True)
        sc = vals[1] if len(vals) > 1 else 0.0
        w.writerow([r['sample_id'], r['dataset'], f'{sc:.4f}', int(sc >= THR), sum(p[0] >= THR for p in keep), f'{vals[0] if vals else 0:.4f}', r['score'],
                    len(r['pk']) - len(keep), r['colour_fraction']])
        c = per_ds[r['dataset']]; c['images'] += 1; c['with_caliper'] += sc >= THR; c['raw_above_threshold'] += float(r['score']) >= THR
        c['in_format_with_static_map'] += len(groups[g]) >= MIN_GROUP
for d, c in per_ds.items():
    summary[d] = dict(c, share_with_caliper=round(c['with_caliper'] / c['images'], 4))
    print(d, summary[d])
json.dump(dict(threshold=THR, cell=CELL, min_group=MIN_GROUP, static_share=STATIC_SHARE,
               static_cells={'|'.join(g): len(c) for g, c in static.items() if c}, datasets=summary), open(OUTS, 'w'), indent=1)
