"""Detector of burned-in measurement calipers ('+' and 'x' marks) in ultrasound frames.

A small fully convolutional network scores every 4th pixel for "a caliper is centred here". It is trained
only on synthetic calipers drawn onto real frames (train_marker_net.py); it never sees a diagnosis label
and no hand label is used for training. The per-image flag is built in finalize_markers.py (two marks required,
fixed scanner furniture removed).

A hand-written cross filter was tried first and dropped: it fired on tissue speckle and on scanner icons
(see PROGRESS.md, 2026-11-23/24).
"""
import numpy as np
import cv2
import torch
import torch.nn as nn

PATCH = 40          # training patch; the label refers to its centre
STRIDE = 4


class MarkerNet(nn.Module):
    def __init__(self, c=24):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.f = nn.Sequential(*block(3, c), *block(c, c), nn.MaxPool2d(2),
                               *block(c, 2 * c), *block(2 * c, 2 * c), nn.MaxPool2d(2),
                               *block(2 * c, 3 * c), *block(3 * c, 3 * c), nn.Conv2d(3 * c, 1, 1))

    def forward(self, x):
        return self.f(x)


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def to_tensor(img_bgr):
    return torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float().div(255.0)


def load(path):
    net = MarkerNet(); net.load_state_dict(torch.load(path, map_location='cpu')); net.eval()
    return net


PAD = 24            # frame is extended by edge replication so that the zero padding of the network stays outside


@torch.no_grad()
def detect(net, img_bgr, max_peaks=20, floor=0.3):
    """Return (score, peaks); peaks are (probability, x, y), strongest first; score is the strongest."""
    big = cv2.copyMakeBorder(img_bgr, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    p = torch.sigmoid(net(to_tensor(big)[None]))[0, 0].numpy()
    m = PAD // STRIDE
    p = p[m:p.shape[0] - m, m:p.shape[1] - m]
    q = p.copy(); peaks = []
    for _ in range(max_peaks):
        y, x = np.unravel_index(int(q.argmax()), q.shape)
        if q[y, x] < floor: break
        peaks.append((float(q[y, x]), int(x * STRIDE + STRIDE // 2), int(y * STRIDE + STRIDE // 2)))
        cv2.circle(q, (int(x), int(y)), 3, 0, -1)
    return float(p.max()), peaks


def colour_fraction(img_bgr):
    """Share of pixels that are clearly coloured (Doppler, coloured overlays)."""
    i = img_bgr.astype(np.int16)
    return float(((i.max(axis=2) - i.min(axis=2)) > 60).mean())
