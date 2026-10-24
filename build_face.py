"""Build a textured GLB head/bust from a single frontal photograph.

Front-facing surface is textured by orthographic projection of the photo (so the frontal view matches the photo
pixel-for-pixel); relief (nose, brow, cheeks, lips, chin) is displaced in depth at positions measured on the photo.
Back/side surfaces use flat swatches sampled from the photo, with light baked into vertex colours.
"""
import json, struct, sys, io
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator

SRC = '/mnt/user-data/attachments/0c41eb21-2ee3-5a90-a254-e65e98147b5b/23bdc3b3-image.png'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'face.glb'
S, XC, YC = 300.0, 348.0, 380.0           # px per world unit, origin px
NC = 88                                   # columns around

photo = Image.open(SRC).convert('RGB')
W, H = photo.size
arr = np.asarray(photo).astype(float)
mean = lambda x0, x1, y0, y1: arr[y0:y1, x0:x1].reshape(-1, 3).mean(0)
SW = [  # swatch colours sampled from the photo
    None,
    mean(165, 185, 290, 330) * 0 + np.array([14, 14, 12]),           # 1 hair
    mean(290, 400, 238, 272) * 0.92,                                  # 2 skin (forehead, slightly shaded)
    mean(100, 200, 700, 740),                                         # 3 jacket
    mean(300, 400, 40, 90),                                           # 4 cap crown
    mean(230, 300, 205, 225),                                         # 5 cap band
]
STRIP = 32
HA = H + STRIP
atlas = Image.new('RGB', (W, HA), (0, 0, 0))
atlas.paste(photo, (0, 0))
px = atlas.load()
for k in range(1, 6):
    c = tuple(int(v) for v in np.clip(SW[k], 0, 255))
    for x in range(int((k - 1) * W / 5), int(k * W / 5)):
        for y in range(H, HA):
            px[x, y] = c
buf = io.BytesIO(); atlas.save(buf, 'JPEG', quality=90, optimize=True); JPG = buf.getvalue()
swuv = lambda k: ((k - 1 + 0.5) / 5.0, (H + STRIP / 2) / HA)

def pch(xs, ys): return PchipInterpolator(xs, ys)

def gauss(x, y, cx, cy, sx, sy, h): return h * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))

def relief(x, y):
    """depth displacement (px) at photo position (x,y); positions measured on the photo."""
    z = 0
    z += gauss(x, y, 349, 250, 100, 50, 10)                    # forehead
    z += gauss(x, y, 272, 314, 45, 12, 14) + gauss(x, y, 422, 314, 45, 12, 14)   # brow ridge
    z += gauss(x, y, 275, 352, 34, 15, -12) + gauss(x, y, 420, 350, 34, 15, -12)  # eye sockets
    z += gauss(x, y, 349, 385, 20, 55, 26)                      # nose bridge
    z += gauss(x, y, 350, 424, 26, 19, 30)                      # nose tip
    z += gauss(x, y, 321, 440, 14, 12, 18) + gauss(x, y, 380, 440, 14, 12, 18)    # alae
    z += gauss(x, y, 350, 462, 22, 9, -8)                       # under nose
    z += gauss(x, y, 250, 448, 52, 45, 22) + gauss(x, y, 448, 448, 52, 45, 22)    # cheeks
    z += gauss(x, y, 355, 487, 46, 13, 8)                      # lips
    z += gauss(x, y, 355, 520, 35, 8, -5)                       # mentolabial groove
    z += gauss(x, y, 350, 556, 45, 24, 10)                      # chin
    return z

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def lathe(ys, Wd, ZF, ZB, zc, xc, NR, bump=False, extra=None, droop=0.0):
    """surface of revolution. rows follow image y; front is +z; seam at back."""
    y = np.linspace(ys[0], ys[-1], NR)
    w, zf, zb = pch(ys, Wd)(y), pch(ys, ZF)(y), pch(ys, ZB)(y)
    th = -np.pi + 2 * np.pi * np.arange(NC) / NC
    s, c = np.sin(th)[None, :], np.cos(th)[None, :]
    X = xc + w[:, None] * s
    Z = zc + np.where(c >= 0, zf[:, None], zb[:, None]) * c
    Y = np.repeat(y[:, None], NC, 1).astype(float)
    if droop:
        Y = Y + droop * ((w[:, None] * s) / 180.0) ** 2
    if bump:
        Z = Z + relief(X, Y) * smooth(0.0, 0.5, c)
    return X, Y, Z, y

def to_world(X, Y, Z): return np.stack([(X - XC) / S, -(Y - YC) / S, Z / S], -1)

def grid_tris(NR):
    t = []
    for i in range(NR - 1):
        for j in range(NC):
            a, b = i * NC + j, i * NC + (j + 1) % NC
            c, d = (i + 1) * NC + j, (i + 1) * NC + (j + 1) % NC
            t += [(a, c, b), (c, d, b)]
    return np.array(t)

def normals(P, T):
    n = np.zeros_like(P)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3): np.add.at(n, T[:, k], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l < 1e-12] = 1
    return n / l

parts = []   # dict(P, N, Nbase, T, pxy, mode_fn)

def add_part(X, Y, Z, NR, mode_of_tri, Xb=None, Yb=None, Zb=None, closed_rows=True):
    P = to_world(X, Y, Z).reshape(-1, 3)
    Pb = to_world(*(Xb, Yb, Zb)).reshape(-1, 3) if Xb is not None else P
    T = grid_tris(NR)
    parts.append(dict(P=P, T=T, N=normals(P, T), Nb=normals(Pb, T), px=np.stack([X.ravel(), Y.ravel()], 1), mode=mode_of_tri))

# ---------------------------------------------------------------- head (skull + face)
hy = np.array([45, 70, 90, 130, 200, 280, 340, 400, 450, 500, 540, 570, 590, 603.])
hw = np.array([0, 60, 105, 150, 168, 174, 172, 162, 146, 118, 93, 64, 32, 0.])
hzf = np.array([0, 45, 85, 135, 172, 186, 190, 190, 182, 165, 140, 100, 50, 0.])
hzb = np.array([0, 60, 115, 170, 205, 215, 210, 190, 160, 120, 80, 50, 25, 0.])
NRh = 110
X, Y, Z, yrow = lathe(hy, hw, hzf, hzb, 0, XC, NRh, bump=True)
Xb, Yb, Zb, _ = lathe(hy, hw, hzf, hzb, 0, XC, NRh, bump=False)
def head_mode(ymean, nzb):
    if nzb > 0.08: return 0
    if ymean < 410: return 1
    if nzb > -0.3: return 6
    return 1 if ymean < 520 else 2
head_w = pch(hy, hw)
add_part(X, Y, Z, NRh, head_mode, Xb, Yb, Zb)

# ---------------------------------------------------------------- cap (crown + band), droop mimics the photo's viewing angle
CXc, ZCc, KZ, D = 338.0, -35.0, 1.22, 70.0
pt = np.arange(9.)
cr = np.array([178, 180, 186, 205, 212, 200, 150, 80, 0.])
cy_ = np.array([215, 168, 118, 100, 80, 45, 20, 12, 10.])
NRc = 64
tt = np.linspace(8, 0, NRc)   # row 0 = crown pole, last row = rim (rows run downward like the head)
r_, yc_ = pch(pt, cr)(tt), pch(pt, cy_)(tt)
th = -np.pi + 2 * np.pi * np.arange(NC) / NC
s, c = np.sin(th)[None, :], np.cos(th)[None, :]
Xc = CXc + r_[:, None] * s
Zc = ZCc + KZ * r_[:, None] * c
Yc = yc_[:, None] + D * ((r_[:, None] * s) / 180.0) ** 2
ycen = np.repeat(yc_[:, None], NC, 1)
Tc = grid_tris(NRc)
def cap_mode(ymean_unused, nzb, tri_idx=None): return 0
Pc = to_world(Xc, Yc, Zc).reshape(-1, 3)
parts.append(dict(P=Pc, T=Tc, N=normals(Pc, Tc), Nb=normals(Pc, Tc), px=np.stack([Xc.ravel(), Yc.ravel()], 1),
                  mode=lambda ym, nzb, ycv=ycen.ravel(): 0, ycen=ycen.ravel(), iscap=True))

# ---------------------------------------------------------------- neck + collar + shoulders
by = np.array([515, 540, 575, 600, 650, 690, 760, 800, 830.])
bw = np.array([80, 110, 128, 138, 260, 335, 340, 325, 0.])
bzf = np.array([70, 90, 102, 108, 120, 130, 130, 120, 0.])
bzb = np.array([60, 70, 78, 80, 100, 110, 105, 95, 0.])
NRb = 70
X, Y, Z, _ = lathe(by, bw, bzf, bzb, -35, XC, NRb)
def body_mode(ymean, nzb):
    if ymean > H - 14: return 3
    return 0 if nzb > 0.08 else (2 if ymean < 548 else 3)
add_part(X, Y, Z, NRb, body_mode)

# ---------------------------------------------------------------- ears
def ellipsoid(cx, cy, cz, rx, ry, rz, nr=14, nc=20):
    v = np.linspace(0.0, np.pi, nr); u = np.linspace(-np.pi, np.pi, nc, endpoint=False)
    V, U = np.meshgrid(v, u, indexing='ij')
    X = cx + rx * np.sin(V) * np.sin(U); Y = cy - ry * np.cos(V); Z = cz + rz * np.sin(V) * np.cos(U)
    P = to_world(X, Y, Z).reshape(-1, 3)
    T = []
    for i in range(nr - 1):
        for j in range(nc):
            a, b = i * nc + j, i * nc + (j + 1) % nc; cc, d = (i + 1) * nc + j, (i + 1) * nc + (j + 1) % nc
            T += [(a, cc, b), (cc, d, b)]
    T = np.array(T)
    return dict(P=P, T=T, N=normals(P, T), Nb=normals(P, T), px=np.stack([X.ravel(), Y.ravel()], 1), mode=lambda ym, nzb: 2)
parts.append(ellipsoid(XC - 172, 366, -10, 11, 38, 26))
parts.append(ellipsoid(XC + 168, 366, -10, 11, 38, 26))

# ---------------------------------------------------------------- assemble with per-triangle texture mode
LDIR = np.array([0.35, 0.6, 0.72]); LDIR /= np.linalg.norm(LDIR)
pos, nor, uv, col, idx = [], [], [], [], []
count = 0
for p in parts:
    P, T, N, Nb, pxy = p['P'], p['T'], p['N'], p['Nb'], p['px']
    cache = {}
    for ti, tri in enumerate(T):
        nzb = Nb[tri, 2].mean(); ym = pxy[tri, 1].mean()
        if p.get('iscap'):
            yc_m = p['ycen'][tri].mean()
            mode = 0 if nzb > 0.08 else (4 if yc_m < 125 else 5)
        else:
            mode = p['mode'](ym, nzb)
        side = (1 if pxy[tri, 0].mean() > XC else -1) if mode == 6 else 0
        for v in tri:
            key = (v, mode, side)
            if key not in cache:
                cache[key] = count; count += 1
                pos.append(P[v]); nor.append(N[v])
                if mode == 0:
                    uv.append((pxy[v, 0] / W, min(pxy[v, 1], H - 2) / HA)); wf = smooth(0.08, 0.5, Nb[v, 2])
                elif mode == 6:   # continue the face's edge pixels outward (matches the seam colour exactly)
                    yy = float(np.clip(pxy[v, 1], 45, 603)); ex = XC + side * (float(head_w(yy)) - 6.0)
                    uv.append((ex / W, min(yy, H - 2) / HA)); wf = 0.55
                else:
                    uv.append(swuv(mode)); wf = 0.0
                shade = 0.50 + 0.50 * max(0.0, float(N[v] @ LDIR)) + 0.12 * max(0.0, float(N[v, 1]))
                shade = min(shade, 1.0)
                f = wf * 1.0 + (1 - wf) * shade
                col.append((f, f, f, 1.0))
            idx.append(cache[key])
pos = np.array(pos, np.float32); nor = np.array(nor, np.float32); uv = np.array(uv, np.float32)
col8 = np.clip(np.array(col) * 255 + 0.5, 0, 255).astype(np.uint8)
idx = np.array(idx, np.uint32)
use16 = len(pos) < 65535
idx = idx.astype(np.uint16 if use16 else np.uint32)

# ---------------------------------------------------------------- write GLB
def pad(b, ch=b'\x00'): return b + ch * ((4 - len(b) % 4) % 4)
chunks = []; views = []; accs = []
def add_view(data, target=None):
    off = sum(len(c) for c in chunks); chunks.append(pad(data))
    v = dict(buffer=0, byteOffset=off, byteLength=len(data))
    if target: v['target'] = target
    views.append(v); return len(views) - 1
vP = add_view(pos.tobytes(), 34962); vN = add_view(nor.tobytes(), 34962); vU = add_view(uv.tobytes(), 34962)
vC = add_view(col8.tobytes(), 34962); vI = add_view(idx.tobytes(), 34963); vJ = add_view(JPG)
accs = [
    dict(bufferView=vP, componentType=5126, count=len(pos), type='VEC3', min=pos.min(0).tolist(), max=pos.max(0).tolist()),
    dict(bufferView=vN, componentType=5126, count=len(pos), type='VEC3'),
    dict(bufferView=vU, componentType=5126, count=len(pos), type='VEC2'),
    dict(bufferView=vC, componentType=5121, normalized=True, count=len(pos), type='VEC4'),
    dict(bufferView=vI, componentType=5123 if use16 else 5125, count=len(idx), type='SCALAR'),
]
gltf = dict(
    asset=dict(version='2.0', generator='mailroom-face3d'),
    extensionsUsed=['KHR_materials_unlit'],
    scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name='head_bust')],
    meshes=[dict(primitives=[dict(attributes=dict(POSITION=0, NORMAL=1, TEXCOORD_0=2, COLOR_0=3), indices=4, material=0, mode=4)])],
    materials=[dict(name='photo', pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0.0, roughnessFactor=0.9),
                    extensions=dict(KHR_materials_unlit={}))],
    textures=[dict(source=0, sampler=0)], samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
    images=[dict(bufferView=vJ, mimeType='image/jpeg')],
    accessors=accs, bufferViews=views, buffers=[dict(byteLength=sum(len(c) for c in chunks))],
)
js = pad(json.dumps(gltf, separators=(',', ':')).encode(), b' ')
bin_ = b''.join(chunks)
glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(bin_), 0x004E4942) + bin_
open(OUT, 'wb').write(glb)
print('verts', len(pos), 'tris', len(idx) // 3, 'bytes', len(glb), 'jpg', len(JPG))
