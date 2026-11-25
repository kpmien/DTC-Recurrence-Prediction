"""Train the caliper detector on synthetic calipers.

Usage: python -I train_marker_net.py <data/external> <manifest.csv> <out_weights.pt>

Backgrounds: training images only (dev_a/train of TN5000, TN3K, ThyroidXL) and ThyUS2Path batch 1, which
takes no part in the study. No diagnosis label and no hand label is read.
Positive patch: a synthetic caliper within 3 px of the patch centre. Negative patch: none near the centre.
Both kinds may carry synthetic distractors (text, ticks, lines, dots, corners, dotted boxes, colour flow patches,
bright specks, off-centre calipers). Version 2 (2026-11-25) adds tiny and faint calipers, stronger degradation,
enlarged ThyroidXL crops as marker-free backgrounds and fixed scanner furniture as hard negatives.
Some backgrounds already contain real calipers; such a patch is then a wrongly labelled negative. With
random 40 px patches this is rare (about 1%) and is accepted.
"""
import sys, os, csv, random, time
import numpy as np, cv2, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers

R, MAN, OUT = sys.argv[1:4]
SEED = 20261123
N_BG = 350            # background images per source
STEPS, BATCH = int(os.environ.get("MARKER_STEPS", 2500)), 128
P = markers.PATCH
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(8)

COLOURS = [(255, 255, 255)] * 5 + [(0, 255, 255), (0, 200, 255), (0, 165, 255), (90, 255, 90), (255, 255, 0), (255, 160, 60)]  # BGR


def draw_caliper(img, cx, cy, rng):
    """Draw one caliper centred at (cx, cy) on a 3x supersampled canvas, then merge."""
    S = 3; h, w = img.shape[:2]
    lay = np.zeros((h * S, w * S, 3), np.uint8); alpha = np.zeros((h * S, w * S), np.uint8)
    col = rng.choice(COLOURS)
    if rng.random() < 0.4: a = rng.uniform(2.0, 4.5); t = max(2, int(round(rng.uniform(0.6, 1.3) * S)))   # tiny marks (small, enlarged crops)
    else: a = rng.uniform(3.0, 9.0); t = max(1, int(round(rng.uniform(1.0, 2.6) * S)))
    op = rng.uniform(0.45, 1.0) if rng.random() < 0.5 else 1.0   # faint marks
    kind = rng.choice(['plus', 'plus', 'plus', 'x', 'open', 'openx'])
    ang = rng.uniform(-4, 4) + (45 if kind in ('x', 'openx') else 0)
    gap = rng.uniform(1.0, 2.2) if kind.startswith('open') else 0
    outline = kind.startswith('open') or rng.random() < 0.15
    segs = []
    for k in range(4):
        th = np.deg2rad(ang + 90 * k); dx, dy = np.cos(th), np.sin(th)
        p0 = (int((cx + dx * gap) * S), int((cy + dy * gap) * S)); p1 = (int((cx + dx * a) * S), int((cy + dy * a) * S))
        segs.append((p0, p1))
    if outline:
        for p0, p1 in segs: cv2.line(alpha, p0, p1, 255, t + 2 * S, cv2.LINE_AA)   # dark halo
    for p0, p1 in segs:
        cv2.line(lay, p0, p1, col, t, cv2.LINE_AA)
    m = cv2.cvtColor(lay, cv2.COLOR_BGR2GRAY)
    alpha = np.maximum(alpha, (m > 0).astype(np.uint8) * 255)
    lay = cv2.resize(lay, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    al = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    al = al * op
    out = img.astype(np.float32) * (1 - al) + lay * op
    img[:] = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:   # number next to the mark
        cv2.putText(img, rng.choice('1234D'), (int(cx + rng.choice([-1, 1]) * rng.uniform(5, 11) - 3), int(cy + rng.choice([-1, 1]) * rng.uniform(4, 9) + 3)),
                    rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN]), rng.uniform(0.3, 0.9), col, 1, cv2.LINE_AA)
    if rng.random() < 0.35:   # dotted line leaving the mark
        th = rng.uniform(0, 2 * np.pi); step = rng.uniform(3, 6)
        for d in np.arange(a + 2, 60, step):
            x, y = int(cx + np.cos(th) * d), int(cy + np.sin(th) * d)
            if 0 <= x < w and 0 <= y < h: img[y, x] = col


def draw_distractor(img, rng, keep_out=None):
    h, w = img.shape[:2]; col = rng.choice(COLOURS)
    x, y = rng.randrange(w), rng.randrange(h)
    if keep_out is not None and abs(x - keep_out[0]) < 12 and abs(y - keep_out[1]) < 12: return
    k = rng.choice(['text', 'text', 'tick', 'line', 'dot', 'dot', 'corner', 'T', 'dotline', 'arrow', 'rect', 'dotrect', 'doppler', 'specks'])
    L = rng.randint(4, 14)
    if k == 'text':
        s = ''.join(rng.choice('0123456789cmDLRxt.+-=:/ABEFHIKTX') for _ in range(rng.randint(1, 5)))
        cv2.putText(img, s, (x, y), rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX]), rng.uniform(0.3, 1.1), col, rng.choice([1, 1, 2]), cv2.LINE_AA)
    elif k == 'tick':   # scale bar with ticks
        cv2.line(img, (x, 0), (x, h), col, 1)
        for yy in range(rng.randrange(8), h, rng.randint(6, 14)): cv2.line(img, (x - rng.randint(0, 4), yy), (x + rng.randint(2, 6), yy), col, 1)
    elif k == 'line':
        th = rng.uniform(0, np.pi); cv2.line(img, (x, y), (int(x + np.cos(th) * 60), int(y + np.sin(th) * 60)), col, rng.choice([1, 1, 2]), cv2.LINE_AA)
    elif k == 'dot':
        cv2.circle(img, (x, y), rng.randint(1, 3), col, -1, cv2.LINE_AA)
    elif k == 'corner':
        cv2.line(img, (x, y), (x + L, y), col, 1); cv2.line(img, (x, y), (x, y + L), col, 1)
    elif k == 'T':
        cv2.line(img, (x - L, y), (x + L, y), col, rng.choice([1, 2])); cv2.line(img, (x, y), (x, y + rng.choice([-1, 1]) * L), col, rng.choice([1, 2]))
    elif k == 'dotline':
        th = rng.uniform(0, np.pi)
        for d in range(0, 60, rng.randint(3, 6)):
            xx, yy = int(x + np.cos(th) * d), int(y + np.sin(th) * d)
            if 0 <= xx < w and 0 <= yy < h: img[yy, xx] = col
    elif k == 'arrow':
        cv2.arrowedLine(img, (x, y), (x + rng.randint(-14, 14), y + rng.randint(-14, 14)), col, 1, cv2.LINE_AA, tipLength=0.4)
    elif k == 'dotrect':   # dotted region box
        x1, y1 = x + rng.randint(15, 60), y + rng.randint(15, 60); st = rng.randint(3, 6)
        for xx in range(x, x1, st):
            for yy in (y, y1):
                if 0 <= xx < w and 0 <= yy < h: img[yy, xx] = col
        for yy in range(y, y1, st):
            for xx in (x, x1):
                if 0 <= xx < w and 0 <= yy < h: img[yy, xx] = col
    elif k == 'doppler':   # colour flow patches
        for _ in range(rng.randint(1, 5)):
            c2 = rng.choice([(0, 0, 230), (230, 40, 0), (255, 120, 0), (0, 90, 255), (0, 220, 255)])
            cv2.ellipse(img, (x + rng.randint(-15, 15), y + rng.randint(-15, 15)), (rng.randint(1, 8), rng.randint(1, 4)), rng.uniform(0, 180), 0, 360, c2, -1, cv2.LINE_AA)
    elif k == 'specks':    # bright echoes (calcifications, comet tails)
        for _ in range(rng.randint(1, 6)):
            g = rng.randint(170, 255); xx, yy = x + rng.randint(-12, 12), y + rng.randint(-12, 12)
            cv2.ellipse(img, (xx, yy), (rng.randint(1, 3), rng.randint(1, 2)), rng.uniform(0, 180), 0, 360, (g, g, g), -1, cv2.LINE_AA)
            if rng.random() < 0.4: cv2.line(img, (xx, yy), (xx + rng.randint(-1, 1), yy + rng.randint(3, 8)), (g // 2, g // 2, g // 2), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x, y), (x + rng.randint(8, 40), y + rng.randint(8, 40)), col, 1)


def degrade(img, rng):
    if rng.random() < 0.5:
        f = rng.uniform(0.4, 1.0); h, w = img.shape[:2]
        img = cv2.resize(cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.6:
        _, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 95)]); img = cv2.imdecode(b, cv2.IMREAD_COLOR)
    return img


def sources():
    rows = [r for r in csv.DictReader(open(MAN)) if r['experimental_split'] == 'dev_a' and r['dev_role'] == 'train']
    rng = random.Random(SEED); out = {}
    for d in ('tn5000', 'tn3k', 'thyroidxl'):
        out[d] = rng.sample([r['data_path'] for r in rows if r['dataset'] == d], N_BG)
    b1 = 'thyus2path/files/batch1_image/dataset'
    out['thyus_b1'] = rng.sample([f'{b1}/{f}' for f in sorted(os.listdir(os.path.join(R, b1)))], N_BG)
    return out


def main():
    rng = random.Random(SEED + 1)
    bgs = []
    for d, ps in sources().items():
        for p in ps:
            im = markers.read(os.path.join(R, p))
            if im is not None and min(im.shape[:2]) >= 2 * P: bgs.append((d, im))
    zr = random.Random(SEED + 2)   # enlarged ThyroidXL crops: marker-free frames with the coarse texture of cropped datasets
    for d, im in [b for b in bgs if b[0] == 'thyroidxl'][:200]:
        h, w = im.shape[:2]; f = zr.uniform(1.6, 2.6); ch, cw = int(h / f), int(w / f)
        y, x = zr.randrange(h - ch), zr.randrange(w - cw)
        bgs.append(('thyroidxl_zoom', cv2.GaussianBlur(cv2.resize(im[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), zr.uniform(0.3, 0.8))))
    weights = np.array([2.0 if d == 'thyroidxl' else 1.0 for d, _ in bgs]); weights /= weights.sum()   # marker-free source counted twice
    print('backgrounds', len(bgs), flush=True)
    W = 2 * P   # work on an 80 px window so that degradation and distractors have context, then cut the centre

    hard = []   # (background index, x, y): places where the current network fires on marker-free frames

    def sample():
        use_hard = hard and rng.random() < 0.35
        if use_hard:
            bi, hx, hy = rng.choice(hard); d, im = bgs[bi]
            h, w = im.shape[:2]
            x = min(max(hx - W // 2 + rng.randint(-2, 2), 0), w - W); y = min(max(hy - W // 2 + rng.randint(-2, 2), 0), h - W)
        else:
            d, im = bgs[np.random.choice(len(bgs), p=weights)]
            h, w = im.shape[:2]; y, x = rng.randrange(h - W), rng.randrange(w - W)
        win = im[y:y + W, x:x + W].copy(); c = W // 2
        pos = rng.random() < (0.3 if use_hard else 0.5)
        cx, cy = c + rng.uniform(-3, 3), c + rng.uniform(-3, 3)
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])): draw_distractor(win, rng, keep_out=(c, c) if not pos else None)
        if pos: draw_caliper(win, cx, cy, rng)
        if rng.random() < 0.5:   # another caliper away from the centre: must not fire at the centre
            ox, oy = rng.choice([-1, 1]) * rng.uniform(13, 30), rng.choice([-1, 1]) * rng.uniform(13, 30)
            draw_caliper(win, c + ox, c + oy, rng)
        win = degrade(win, rng)
        o = (W - P) // 2
        return markers.to_tensor(win[o:o + P, o:o + P]), float(pos)

    net = markers.MarkerNet(); lossf = nn.BCEWithLogitsLoss()
    c = (P // markers.STRIDE) // 2; t0 = time.time()

    def fit(steps, lr, tag):
        net.train(); opt = torch.optim.AdamW(net.parameters(), lr / 2, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps); run = []
        for step in range(1, steps + 1):
            xs, ys = zip(*(sample() for _ in range(BATCH)))
            x = torch.stack(xs); y = torch.tensor(ys)
            logit = net(x)[:, 0, c, c]
            loss = lossf(logit, y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run.append(((logit > 0).float() == y).float().mean().item())
            if step % 250 == 0:
                print(f'{tag} step {step} loss {loss.item():.4f} acc(last 250) {np.mean(run[-250:]):.4f} sec {time.time() - t0:.0f}', flush=True)

    def mine():
        """Collect firing locations on ThyroidXL training frames. The dataset paper states, and the frames looked
        at confirm, that ThyroidXL carries no calipers; the hand-labelled check sample tests this assumption."""
        net.eval(); found = []; n_img = 0
        for bi, (d, im) in enumerate(bgs):
            if d.startswith('thyroidxl'):
                _, pk = markers.detect(net, im, max_peaks=30, floor=0.2)
                n_img += bool(pk and pk[0][0] >= 0.5)
                found += [(bi, x, y) for v, x, y in pk]
        print(f'mining: {len(found)} locations; ThyroidXL frames with a peak >= 0.5: {n_img} of {sum(d.startswith("thyroidxl") for d, _ in bgs)}', flush=True)
        # Fixed scanner furniture in the other sources: places where the network fires at the same position in
        # at least 8% of the frames of one format (30 frames or more). Calipers do not repeat in position.
        groups = {}
        for bi, (d, im) in enumerate(bgs):
            if d in ('tn5000', 'thyus_b1'): groups.setdefault((d, im.shape[:2]), []).append(bi)
        nf = 0
        for g, bis in groups.items():
            if len(bis) < 30: continue
            pks = {bi: markers.detect(net, bgs[bi][1], max_peaks=30, floor=0.3)[1] for bi in bis}; cnt = {}
            for bi, pk in pks.items():
                for cell in {(x // 16, y // 16) for v, x, y in pk}: cnt[cell] = cnt.get(cell, 0) + 1
            hot = {cell for cell, n in cnt.items() if n >= 0.08 * len(bis)}
            for bi, pk in pks.items():
                for v, x, y in pk:
                    if (x // 16, y // 16) in hot: found.append((bi, x, y)); nf += 1
        print(f'mining: {nf} furniture locations in {sum(len(b) >= 30 for b in groups.values())} formats', flush=True)
        return found

    init = os.environ.get('MARKER_INIT')   # continue from saved weights with further hard-negative rounds
    if init: net.load_state_dict(torch.load(init, map_location='cpu'))
    else: fit(STEPS, 4e-3, 'stage1')
    for rnd in range(1, int(os.environ.get('MARKER_ROUNDS', 3)) + 1):
        hard += mine()
        fit(STEPS // 2, 2e-3, f'hard{rnd}')
    mine()
    net.eval()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(net.state_dict(), OUT); print('saved', OUT, 'parameters', sum(p.numel() for p in net.parameters()))


if __name__ == '__main__':
    main()
