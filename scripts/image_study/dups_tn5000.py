import sys, os, hashlib, collections, glob
import xml.etree.ElementTree as ET
D = sys.argv[1]
where = {}
for s in ('train', 'val', 'test'):
    for l in open(os.path.join(D, 'ImageSets', 'Main', s + '.txt')): where[l.strip()] = s
g = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(D, 'JPEGImages', '*.jpg'))):
    g[hashlib.sha256(open(p, 'rb').read()).hexdigest()].append(os.path.splitext(os.path.basename(p))[0])
def ann(k):
    o = ET.parse(os.path.join(D, 'Annotations', k + '.xml')).getroot().find('object')
    return o.findtext('name'), tuple(int(o.find('bndbox').findtext(t)) for t in ('xmin', 'ymin', 'xmax', 'ymax'))
groups = [v for v in g.values() if len(v) > 1]
print('duplicate groups', len(groups), 'sizes', collections.Counter(len(v) for v in groups), 'images involved', sum(len(v) for v in groups), 'surplus files', sum(len(v) - 1 for v in groups))
cross = [v for v in groups if len({where[k] for k in v}) > 1]
tt = [v for v in groups if 'test' in {where[k] for k in v} and {where[k] for k in v} & {'train', 'val'}]
print('groups spanning >1 split', len(cross), '| spanning test and train/val', len(tt), '| test images with a byte-identical copy in train/val', sum(where[k] == 'test' for v in tt for k in v))
same_box = diff_label = consec = 0
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u
ious = []
for v in groups:
    A = [ann(k) for k in v]
    if len({a[1] for a in A}) == 1: same_box += 1
    if len({a[0] for a in A}) > 1: diff_label += 1
    if max(map(int, v)) - min(map(int, v)) == len(v) - 1: consec += 1
    ious.append(iou(A[0][1], A[1][1]))
print('groups with identical boxes', same_box, '| groups with conflicting labels', diff_label, '| groups with consecutive IDs', consec)
import statistics
print('box IoU between first two copies: median', round(statistics.median(ious), 3), 'n with IoU==0', sum(i == 0 for i in ious), 'n with IoU>0.5', sum(i > 0.5 for i in ious))
for v in groups[:5]: print(v, [where[k] for k in v], [ann(k) for k in v])
