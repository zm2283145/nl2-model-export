#!/usr/bin/env python3
"""NL2ModelExport - turn one or more captures into model files.

    py nl2_export.py <capture_dir> [<capture_dir> ...] -o <out_dir> [--no-scenery] [--no-terrain] [--format glb|obj|both]

Files are .glb (binary glTF, opens in Blender/Windows 3D viewer, much smaller than OBJ) unless --format obj/both.

Writes into <out_dir>:
    layout.obj / layout.mtl   the coaster as placed in the park, grouped:
                                track      - rails/spine/ties (the engine's world-space track meshes)
                                structure  - supports, footers, catwalks, lift/brake hardware, station, buildings near the track
                                trains     - trains where they were standing when captured
    layout_no_trains.obj      the same without the trains (just track + structure)
    scenery.obj / .mtl        everything else, same coordinates as layout.obj:
                                scenery    - trees, props, buildings away from the ride (omit with --no-scenery)
                                terrain    - ground meshes                           (omit with --no-terrain)
    train_front_car.obj       the front car (with nose) on its own, in its local space, NOT attached to the track
    train_car.obj             a middle car the same way (origin = centre of the car, floor at y=0)
    train_rear_car.obj        the last car (with the rear bumper); all three share the middle car's origin and
                              axes (forward = +X), so cars line up when placed one car spacing apart (see report)
                              (--all-cars also writes train_car1..N.obj)
    report.txt                what was found

How positions are recovered (see NOTES.md): every draw's vertex-shader constant buffer 1 holds its
model-view-projection matrix. The track is drawn in world space, so the track's matrix is the pure
view-projection of each render pass (camera, shadow cascades, reflection). For any other draw,
world = inverse(pass view-projection) x its MVP; we keep the pass that yields a rigid transform.
Merging several captures (different camera positions) fills in parts the game did not draw in one frame.
Only needs numpy.
"""
import argparse, collections, json, os, struct, sys
import numpy as np

# ------------------------------------------------------------------------------------------ capture reading
class Capture:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(os.path.join(path, "draws.json")))
        try: self.n3 = json.load(open(os.path.join(path, "nl3d_buffers.json")))
        except Exception: self.n3 = {}
        raw = open(os.path.join(path, "cbuffers.bin"), "rb").read()
        n = struct.unpack_from("<I", raw, 0)[0]; o = 4; self.cbs = []
        for _ in range(n):
            z = struct.unpack_from("<I", raw, o)[0]; o += 4; self.cbs.append(raw[o:o + z]); o += z
        self._buf = {}; self._mesh = {}; self._uv2 = {}; self._texcache = {}
        try: self.tex = json.load(open(os.path.join(path, "textures.json")))   # v8+ captures only
        except Exception: self.tex = {}

    def buf(self, name):
        if name not in self._buf:
            p = os.path.join(self.path, "buffers", name + ".bin")
            self._buf[name] = np.fromfile(p, dtype=np.uint8) if os.path.exists(p) else None
        return self._buf[name]

    def layout(self, vb):
        info = self.n3.get(vb); pos, nrm, uv, uv2 = 0, None, None, None
        if info and info.get("vb"):
            n3 = 0
            for t, o in zip(info["types"], info["offsets"]):
                if t == 3:
                    if n3 == 0: pos = o
                    elif n3 == 1: nrm = o
                    n3 += 1
                elif t == 2:
                    if uv is None: uv = o
                    elif uv2 is None: uv2 = o
        return pos, nrm, uv, uv2

    # --- textures (v8+ captures): <capture>/textures/<resource>.dds, top mip, decoded on demand
    def texture(self, name):
        """(rgba uint8 array, info dict) or None"""
        if not name or name == "0000000000000000" or name not in self.tex: return None
        if name not in self._texcache:
            r = None
            try:
                import nl2_textures
                d = nl2_textures.decode(os.path.join(self.path, "textures", name + ".dds"))
                if d is not None: r = (d[0], self.tex[name])
            except Exception: r = None
            self._texcache[name] = r
        return self._texcache[name]

    def tex_kind(self, name):
        """swatch (1x1 / 8x1 colour table) | normal | mask | color | None"""
        info = self.tex.get(name)
        if not info: return None
        if info["w"] * info["h"] <= 8: return "swatch"
        if info.get("bind", 8) & 0x20: return None                   # render target (screen, shadows)
        if info["format"] in (79, 80, 81): return "mask"
        if info["format"] in (49, 82, 83, 84): return "normal"       # 49 = R8G8 (2-channel normal map)
        t = self.texture(name)
        if t is None: return None
        m = t[0][::max(1, t[0].shape[0] // 32), ::max(1, t[0].shape[1] // 32), :3].reshape(-1, 3).mean(0)
        if m[2] > 200 and 100 < m[0] < 160 and 100 < m[1] < 160: return "normal"   # tangent-space normal map
        return "color"

    def is_screen_pass(self, i):
        """post-processing / full-screen passes read the rendered image (a window-sized render target) in
        slot 0; their quads would otherwise end up as stray planes in the scene"""
        tex = self.d[i].get("tex")
        info = self.tex.get(tex[0]) if tex else None
        return bool(info and info.get("bind", 8) & 0x20 and info["w"] != info["h"])

    def generic_textures(self):
        """textures bound for a large share of all meshes are the engine's generic overlays (reflection-probe /
        grime passes draw everything with them); they are never an object's own look"""
        if not hasattr(self, "_generic"):
            use = collections.defaultdict(set); meshes = set()
            for x in self.d:
                m = (x["vb"][0], x["ib"], x["startIndex"]); meshes.add(m); use[x["tex"][0] if x.get("tex") else ""].add(m)
            self._generic = {t for t, ms in use.items() if len(ms) > 0.12 * len(meshes) and self.tex.get(t, {}).get("w", 1) * self.tex.get(t, {}).get("h", 1) > 8}
        return self._generic

    def used_slots(self, ps):
        """texture slots a pixel shader really reads. D3D keeps old bindings in slots a shader ignores, so
        v9+ captures record which slots the game re-bound before each draw; the union over a shader's draws
        is what it uses. 0xff = unknown (older capture, or the game never re-bound anything for it)."""
        if not hasattr(self, "_used"):
            self._used = collections.defaultdict(int)
            for x in self.d:
                if "texSet" in x: self._used[x["ps"]] |= x["texSet"] & 0xff
        return self._used.get(ps) or 0xff

    def generic_shaders(self):
        """pixel shaders that draw a large share of all meshes are the shadow / reflection-probe / overlay passes;
        the main view uses a separate shader per material type"""
        if not hasattr(self, "_gshaders"):
            m = collections.defaultdict(set); allm = set()
            for x in self.d:
                k = (x["vb"][0], x["ib"], x["startIndex"]); m[x["ps"]].add(k); allm.add(k)
            self._gshaders = {ps for ps, v in m.items() if len(v) > 0.3 * len(allm)}
        return self._gshaders

    def material(self, i):
        """what a draw looks like: (base texture name or None, colour from a 1x1 in slot 0 or None,
        paint colour (slot 3, 1x1) or None, paint ramp (slot 2, 8x1) or None).
        NL2's coaster shaders: slot 0 = detail texture whose alpha is the rust/dirt amount (0 = fully painted),
        slot 2 = paint: a 1x1 colour, or an 8-texel ramp (track: sampled by the 2nd UV around the cross-section).
        (slot 3 holds a second colour whose use we don't know; it is not the main paint.)"""
        tex = self.d[i].get("tex")
        if not tex: return None
        if self.d[i]["ps"] in self.generic_shaders(): return None
        used = self.used_slots(self.d[i]["ps"])
        tex = [t if used >> k & 1 else "0000000000000000" for k, t in enumerate(tex)]
        if tex[0] in self.generic_textures(): return None
        base = solid = paint = palette = None
        k0 = self.tex_kind(tex[0])
        if k0 == "color": base = tex[0]
        elif k0 == "swatch":
            t = self.texture(tex[0])
            if t is not None: solid = t[0][0, 0, :3].astype(np.float32) / 255.0
        for s_ in (2, 3):
            if self.tex_kind(tex[s_]) != "swatch": continue
            t = self.texture(tex[s_])
            if t is None: continue
            if s_ == 2 and t[0].shape[1] == 8: palette = t[0][0, :, :3].astype(np.float32) / 255.0
            elif s_ == 2 and t[0].shape[1] == 1: paint = t[0][0, 0, :3].astype(np.float32) / 255.0
        # normal-mapped coaster shaders (detail, rust mask or 2nd normal map, normal map, paint): the paint (a 1x1
        # colour, or the 8x1 ramp on brake fins) moves to slot 3. With a paint slot in slot 2, slot 3 is a leftover
        # binding (the same blue on everything) and is ignored
        if paint is None and palette is None and self.tex_kind(tex[1]) in ("mask", "normal") \
                and self.tex_kind(tex[2]) == "normal" and self.tex_kind(tex[3]) == "swatch":
            t = self.texture(tex[3])
            if t is not None and t[0].shape[1] == 1: paint = t[0][0, 0, :3].astype(np.float32) / 255.0
            elif t is not None and t[0].shape[1] == 8: palette = t[0][0, :, :3].astype(np.float32) / 255.0
        # object colour (train parts, props): pixel-shader constant buffer 1 starts with a linear RGB + 1.0
        tint = None
        b = self.d[i].get("pscb", [-1, -1])[1] if self.d[i].get("pscb") else -1
        if 0 <= b < len(self.cbs) and len(self.cbs[b]) >= 16:
            f = np.frombuffer(self.cbs[b][:16], np.float32)
            if f[3] == 1.0 and np.all((f[:3] >= 0) & (f[:3] <= 1)) and f[1] != 256: tint = f[:3].astype(np.float32)
        return base, solid, paint, palette, tint, (base in self.painted_textures())

    def painted_textures(self):
        """detail textures the game draws with a paint colour somewhere in this capture"""
        if not hasattr(self, "_painted"):
            self._painted = set()
            for x in self.d:
                t = x.get("tex")
                if not t: continue
                used = self.used_slots(x["ps"])
                if used >> 2 & 1 and self.tex.get(t[2], {}).get("w", 99) * self.tex.get(t[2], {}).get("h", 99) <= 8:
                    self._painted.add(t[0])
        return self._painted

    def look_score(self, i):
        """higher = more likely the main-view draw (which binds the paint textures too)"""
        tex = self.d[i].get("tex") or []
        if self.d[i]["ps"] in self.generic_shaders(): return -1
        bound = sum(1 for t in tex if t and t != "0000000000000000")
        own = 0
        if tex and tex[0] not in self.generic_textures():
            k = self.tex_kind(tex[0])
            own = 2 if k == "color" else (1 if k == "swatch" else 0)   # a real texture beats a plain 1x1 colour
        return own * 10 + bound

    def uv2(self, i):
        self.mesh(i); return self._uv2.get(i)

    def mesh(self, i):
        if i in self._mesh: return self._mesh[i]
        self._mesh[i] = r = self._mesh_uncached(i)
        return r

    def _mesh_uncached(self, i):
        x = self.d[i]; vb = x["vb"][0]; st = x["stride"][0]; b = self.buf(vb)
        if b is None or not st: return None
        if x["kind"] in (0, 2):
            ib = self.buf(x["ib"])
            if ib is None: return None
            fmt = np.uint32 if x["ibFormat"] == 42 else np.uint16; sz = np.dtype(fmt).itemsize
            a = x["ibOffset"] + x["startIndex"] * sz
            idx = np.frombuffer(ib[a:a + x["count"] * sz].tobytes(), dtype=fmt).astype(np.int64) + x["baseVertex"]
        else:
            idx = np.arange(x["startVertex"], x["startVertex"] + x["count"])
        if x["topology"] == 4:
            tri = idx[:len(idx) // 3 * 3].reshape(-1, 3)
        elif x["topology"] == 5:
            t = [(idx[k], idx[k + 1], idx[k + 2]) if k % 2 == 0 else (idx[k + 1], idx[k], idx[k + 2]) for k in range(len(idx) - 2)]
            t = [q for q in t if len(set(q)) == 3]
            tri = np.array(t, dtype=np.int64) if t else np.zeros((0, 3), np.int64)
        else:
            return None
        if len(tri) == 0: return None
        used, inv = np.unique(tri, return_inverse=True)
        F = inv.reshape(-1, 3)
        offs = x["offset"][0] + used * st
        if offs.max() + st > len(b) or offs.min() < 0: return None
        rows = b[(offs[:, None] + np.arange(st)[None, :])]
        pos, nrm, uv, uv2 = self.layout(vb)
        if uv2 is not None and uv2 + 8 <= st: self._uv2[i] = rows[:, uv2:uv2 + 8].copy().view(np.float32).reshape(-1, 2)
        V = rows[:, pos:pos + 12].copy().view(np.float32).reshape(-1, 3)
        N = rows[:, nrm:nrm + 12].copy().view(np.float32).reshape(-1, 3) if nrm is not None and nrm + 12 <= st else None
        UV = rows[:, uv:uv + 8].copy().view(np.float32).reshape(-1, 2) if uv is not None and uv + 8 <= st else None
        if not np.isfinite(V).all(): return None
        return V, N, UV, F

    def cbf(self, i, slot=1):
        b = self.d[i]["cb"][slot]
        return np.frombuffer(self.cbs[b], dtype=np.float32) if b >= 0 else None

# ------------------------------------------------------------------------------------------ transform recovery
def clip_fraction(P, M):
    H = np.c_[P, np.ones(len(P))]; C = H @ M.T; w = C[:, 3]
    ok = (w > 1e-3) & (np.abs(C[:, 0]) <= w * 1.2) & (np.abs(C[:, 1]) <= w * 1.2) & (C[:, 2] >= -0.1 * w) & (C[:, 2] <= w * 1.1)
    return ok.mean()

def mvp_offsets(cap, sample=25):
    """per vertex shader: float offset of its MVP (4 rows) inside constant buffer 1."""
    g = collections.defaultdict(list)
    for i, x in enumerate(cap.d):
        if x["cb"][1] >= 0: g[x["vs"]].append(i)
    out = {}
    for vs, ids in g.items():
        score = collections.defaultdict(list)
        for i in ids[::max(1, len(ids) // sample)][:sample]:
            m = cap.mesh(i); f = cap.cbf(i)
            if m is None or f is None: continue
            V = m[0][::max(1, len(m[0]) // 40)]
            for off in range(0, len(f) - 15, 4):
                M = f[off:off + 16].reshape(4, 4)
                if not np.isfinite(M).all() or np.abs(M).max() == 0 or abs(np.linalg.det(M.astype(np.float64))) < 1e-12:
                    score[off].append(0); continue
                score[off].append(clip_fraction(V, M))
        if score:
            ranked = sorted(score, key=lambda k: -np.mean(score[k]))
            best = np.mean(score[ranked[0]])
            # keep every plausible offset: a shader can be sampled while its geometry is mostly off screen
            out[vs] = [k for k in ranked if np.mean(score[k]) >= max(0.05, 0.3 * best)][:4] or ranked[:1]
    return out

def mvp_of(cap, offs, i, o=None):
    if o is None:
        L = offs.get(cap.d[i]["vs"]); o = L[0] if L else None
    f = cap.cbf(i)
    if o is None or f is None or o + 16 > len(f): return None
    return f[o:o + 16].reshape(4, 4).astype(np.float64)

def is_rigid(W):
    R = W[:3, :3]; s = np.linalg.norm(R, axis=0)
    if not (abs(W[3, :3]).max() < 1e-2 and abs(W[3, 3] - 1) < 1e-2 and s.min() > 1e-4 and s.max() < 1e4): return False
    Rn = R / s
    return np.allclose(Rn @ Rn.T, np.eye(3), atol=3e-2)

def resolve_worlds(cap, offs):
    # pass matrices = MVPs of draws whose geometry is clearly in world space (world = identity)
    Ms, cnt, index = [], [], {}
    for i in range(len(cap.d)):
        m = cap.mesh(i)
        if m is None or np.linalg.norm(m[0].mean(0)) < 30: continue
        for o in offs.get(cap.d[i]["vs"], []):
            M = mvp_of(cap, offs, i, o)
            if M is None or abs(np.linalg.det(M)) < 1e-14: continue
            q = 1e-4 * max(1.0, float(np.abs(M).max()))
            key = tuple(np.round(M.ravel() / q).astype(np.int64))
            k = index.get(key)
            if k is None: index[key] = len(Ms); Ms.append(M); cnt.append(1)
            else: cnt[k] += 1
    VPs = [M for M, n in sorted(zip(Ms, cnt), key=lambda t: -t[1]) if n >= 3]
    inv = [np.linalg.inv(V) for V in VPs]
    Ws = {}
    for i in range(len(cap.d)):
        for o in offs.get(cap.d[i]["vs"], []):
            M = mvp_of(cap, offs, i, o)
            if M is None: continue
            W = next((iv @ M for iv in inv if is_rigid(iv @ M)), None)
            if W is not None: Ws[i] = W; break
    return Ws

# ------------------------------------------------------------------------------------------ scene assembly
class Part:
    __slots__ = ("V", "N", "UV", "F", "W", "local", "localN", "stride", "key", "group", "mesh_key", "C", "M", "S")

# decoded textures shared by all parts: id (content hash) -> rgba uint8 array
TEXTURES = {}

def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)

def _bake(rgba, paint):
    """paint a detail texture the way NL2's coaster shader does: alpha = rust/dirt amount over the paint colour.
    Textures without such a mask are tinted by the paint colour, keeping their light/dark detail."""
    rgb = rgba[..., :3].astype(np.float32) / 255.0; a = rgba[..., 3:4].astype(np.float32) / 255.0
    if a.mean() < 0.98:
        out = paint * (1 - a) + rgb * a
    else:
        l = rgb.mean(-1, keepdims=True); out = paint * np.clip(0.55 + 0.45 * l / max(float(l.mean()), 1e-3), 0, 1.3)
    o = np.empty_like(rgba); o[..., :3] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8); o[..., 3] = 255
    return o

def _main_colour(palette):
    """colour of a paint ramp for meshes without the 2nd UV: the entry filling most of it (brake fins: one lime
    accent, seven blue), else the first"""
    u, n = np.unique(np.round(palette * 255).astype(int), axis=0, return_counts=True)
    return (u[n.argmax()] / 255.0).astype(np.float32) if n.max() >= 4 else palette[0]

def part_material(cap, i, V):
    """(M, C): M = (texture id or None, (r, g, b)), C = optional per-vertex colours (n, 3) for the painted track"""
    m = cap.material(i)
    if m is None: return None, None
    base, solid, paint, palette, tint, paintable = m
    import hashlib
    C = None
    if palette is not None:
        uv2 = cap.uv2(i)
        if uv2 is not None and len(uv2) == len(V):
            # the 2nd UV runs around the track's cross-section (0 = underside, 1 = top); the 8-texel ramp is
            # sampled with filtering, so interpolate between texel centres
            x = np.clip(uv2[:, 0] * 8 - 0.5, 0, 7); k = np.floor(x).astype(int); f = (x - k)[:, None]
            C = (palette[k] * (1 - f) + palette[np.minimum(k + 1, 7)] * f).astype(np.float32)
    painted = paint is not None or C is not None
    if paint is None: paint = np.ones(3, np.float32) if C is not None else (_main_colour(palette) if palette is not None else None)
    tid, factor = None, (solid if solid is not None else np.ones(3, np.float32))
    if base is not None:
        t = cap.texture(base)
        if t is not None:
            img = t[0]
            if paint is None and paintable and img[..., 3].mean() < 128 and _alpha_mode_arr(img) is None:
                paint = np.ones(3, np.float32); painted = True     # rust/dirt-masked texture whose paint slot wasn't seen: plain white paint
            # a texture without a rust/dirt mask in its alpha is the object's finished look (station roofs, umbrellas,
            # walls: scenery shader); the 1x1 in its paint slot is something else (grey / leftover track blue / black)
            if img[..., 3].mean() >= 0.98 * 255: paint = None if palette is None else paint
            if paint is not None and (painted or palette is not None):
                src = hashlib.md5(img.tobytes()).hexdigest()[:12]
                tid = "p" + hashlib.md5((src + str(np.round(paint, 3).tolist())).encode()).hexdigest()[:11]
                if tid not in TEXTURES: TEXTURES[tid] = _bake(img, paint)
            else:
                tid = hashlib.md5(img.tobytes()).hexdigest()[:12]
                TEXTURES.setdefault(tid, img)
    elif paint is not None and C is None:
        factor = factor * paint
    # factors and vertex colours are linear in glTF/Blender; swatch colours are sRGB bytes
    factor = srgb_to_linear(np.asarray(factor, np.float32))
    if tint is not None: factor = factor * tint
    if C is not None: C = srgb_to_linear(C)
    return (tid, tuple(float(v) for v in np.round(factor, 4))), C

def instance_worlds(cap, i):
    """world matrices of an instanced draw (bolts, nuts, support footers, plants...): every instance record in
    vertex slot 1 starts with a 3x4 world matrix (rows = rotation | translation); the draw's own constant
    buffer then only holds the pass view-projection"""
    x = cap.d[i]; b = x.get("inst", -1); st = x["stride"][1] if len(x["stride"]) > 1 else 0
    if b is None or b < 0 or b >= len(cap.cbs) or st < 48 or st % 4: return []
    raw = cap.cbs[b]; n = min(max(1, x["instances"]), len(raw) // st)
    if n <= 0: return []
    f = np.frombuffer(raw[:n * st], np.float32).reshape(n, st // 4)[:, :12].astype(np.float64)
    out = []
    for r in f:
        W = np.eye(4); W[:3, :] = r.reshape(3, 4)
        if np.isfinite(W).all() and is_rigid(W) and np.abs(W[:3, 3]).max() < 1e5: out.append(W)
    return out

def placements(cap, Ws):
    """(draw, world) for every object drawn: one per plain draw, one per instance of an instanced draw.
    cap.skip_inst: leave this capture's instanced draws out (they come from a harvest, see harvest_instances.py);
    cap.harvest {mesh key: (n,3,4) worlds}: this capture only supplies instanced meshes, and the first draw of each
    harvested mesh gets every harvested world (a superset of its own instances)"""
    harvest = getattr(cap, "harvest", None)
    if harvest is not None:   # hang the harvested worlds on the draw of each mesh with the real look (main view, not shadow)
        best = {}
        for i, x in enumerate(cap.d):
            if x["kind"] in (2, 3) and x.get("inst", -1) >= 0:
                mk = (x["vb"][0], x["ib"], x["startIndex"], x["count"], x["baseVertex"])
                if mk in harvest and (mk not in best or cap.look_score(i) > cap.look_score(best[mk])): best[mk] = i
    for i, x in enumerate(cap.d):
        if x["kind"] in (2, 3) and x.get("inst", -1) >= 0:
            if getattr(cap, "skip_inst", False): continue
            mk = (x["vb"][0], x["ib"], x["startIndex"], x["count"], x["baseVertex"])
            if harvest is not None and mk in harvest:
                if best[mk] != i: continue
                for r in harvest[mk]:
                    W = np.eye(4); W[:3, :] = r; yield i, W
                continue
            for W in instance_worlds(cap, i): yield i, W
        elif i in Ws:
            yield i, Ws[i]

def collect_parts(caps, max_extent=600.0, log=print):
    parts, placed = [], {}
    for cap in caps:
        # every object is drawn several times per frame (shadow maps, reflection cube map, mirror, main view);
        # keep the look of the pass that gives it its own texture / paint colour (see Capture.generic_textures)
        if getattr(cap, "harvest", None) is not None: Ws = {}   # instanced parts only
        else: Ws = resolve_worlds(cap, mvp_offsets(cap))
        n_inst = sum(1 for i, W in placements(cap, {}))
        log(f"{cap.path}: {len(cap.d)} draws, {len(Ws)} placed, {n_inst} instances")
        mats = {}
        def mat(i, V):
            if i not in mats: mats[i] = part_material(cap, i, V)
            return mats[i]
        for i, W in placements(cap, Ws):
            x = cap.d[i]
            if cap.is_screen_pass(i): continue
            mk = (x["vb"][0], x["ib"], x["startIndex"], x["count"], x["baseVertex"])
            key = mk + tuple(np.round(W[:3, :].ravel(), 2))
            # the same object is drawn in several passes (main, mirror, shadow) and captures; the recovered
            # matrices then differ by a few cm, so compare with a tolerance instead of exact keys.
            # Instanced draws (bolts, nuts, footers) carry exact matrices: only an exact repeat is the same object
            # (bolts on a flange sit a few cm apart), and two instances of one draw are never the same object
            inst = x["kind"] in (2, 3) and x.get("inst", -1) >= 0
            draw = (id(cap), i) if inst else None
            tol = 0.005 if inst else 0.2
            cell = tuple(np.floor(W[:3, 3] / 0.2).astype(int))
            near = [Q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    for Q in placed.get(mk + (cell[0] + dx, cell[1] + dy, cell[2] + dz), ())]
            near = [(float(np.abs(W[:3, 3] - Q[0][:3, 3]).max()), n, Q) for n, Q in enumerate(near)
                    if draw not in Q[2] and np.abs(W[:3, :3] - Q[0][:3, :3]).max() < 0.03]
            dup = min(((d, n, Q) for d, n, Q in near if d < tol), default=(0, 0, None))[2]
            if dup is not None:
                if draw: dup[2].add(draw)
                # shadow/depth passes draw the same object without textures: take the look from a pass that has one
                q = dup[1]
                if q is not None and cap.tex and cap.look_score(i) > q.S:
                    M, C = mat(i, q.local)
                    if M is not None: q.M, q.C, q.S = M, C, cap.look_score(i)
                continue
            entry = [W, None, {draw} if draw else set()]; placed.setdefault(mk + cell, []).append(entry)
            m = cap.mesh(i)
            if m is None: continue
            V, N, UV, F = m
            R = W[:3, :3]; t = W[:3, 3]
            Vw = (V @ R.T + t).astype(np.float32)
            ext = Vw.max(0) - Vw.min(0)
            if ext.max() > max_extent: continue
            p = Part()
            p.V = Vw; p.UV = UV; p.F = F; p.W = W; p.local = V; p.localN = N; p.stride = x["stride"][0]
            p.N = (N @ (R / np.linalg.norm(R, axis=0, keepdims=True)).T).astype(np.float32) if N is not None else None
            p.key = key; p.mesh_key = mk + (len(V),); p.group = None
            p.M, p.C = mat(i, V) if cap.tex else (None, None)
            p.S = cap.look_score(i) if p.M is not None else -1
            entry[1] = p
            parts.append(p)
        cap._buf.clear(); cap._mesh.clear(); cap._uv2.clear(); cap._texcache.clear(); cap.cbs = []   # keep memory flat when merging many captures
    return drop_coarse_lods(parts, log)

def drop_coarse_lods(parts, log=print):
    """the same object can come in at several detail levels (distance LODs, shadow meshes, other captures):
    where two different meshes fill the same box, keep the finer one. The tolerance is capped: a long mesh (the
    lift's spine and chain trough, 137 m) is not a detail level of the catwalk beside it just because both are
    137 m long. A coarser level covers about the same surface as the finer one: a solid walkway panel (2 quads)
    is not a detail level of the thin open frame around it (more triangles, a fraction of the area).
    Painted track is left to clean_track."""
    cell = 0.5; grid = collections.defaultdict(list); keep = []
    def area(p):
        V = p.V.astype(np.float64); F = p.F
        return float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2)
    info = sorted(((p, p.V.mean(0), p.V.max(0) - p.V.min(0), area(p)) for p in parts), key=lambda t: -len(t[0].F))
    dropped = 0
    for p, c, e, a in info:
        if p.C is not None: keep.append(p); continue
        k = tuple((c // cell).astype(int)); hit = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q, c2, e2, a2 in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        tol = min(0.03 * e2.max(), 0.3)
                        if q.mesh_key != p.mesh_key and len(p.F) < 0.7 * len(q.F) and a < 1.5 * a2 + 0.05 and \
                           np.abs(c - c2).max() < 0.02 + tol and np.abs(e - e2).max() < tol + 0.01:
                            hit = True; break
                    if hit: break
                if hit: break
            if hit: break
        if hit: dropped += 1; continue
        grid[k].append((p, c, e, a)); keep.append(p)
    if dropped: log(f"dropped {dropped} coarser duplicates")
    return keep

def drop_bad_placements(parts, log=print):
    """remove copies whose world matrix came from the wrong pass:
    - the ground / water reflection pass draws everything mirrored about y = 0 (upside-down supports underground)
    - meshes whose position could not be recovered come out at (or a metre or two from) the world origin
      without rotation; any other placement of the same mesh at the origin is suspect too"""
    def jitter(W):
        return np.abs(W[:3, :3] - np.eye(3)).max() < 0.02 and 1e-3 < np.abs(W[:3, 3]).max() < 4
    bad_mesh = {p.mesh_key for p in parts if jitter(p.W)}
    placed_away = {p.mesh_key for p in parts if np.abs(p.W[:3, 3]).max() >= 4}
    def at_origin(p):
        if np.abs(p.W[:3, 3]).max() >= 4: return False
        if p.mesh_key in bad_mesh: return True
        lo, hi = p.local.min(0), p.local.max(0); c = (lo + hi) / 2
        if identity(p.W) and p.mesh_key in placed_away and np.abs(p.local.mean(0)[[0, 2]]).max() < 30: return True
        # an object's own mesh (centred on its origin) that never got a position; editor gizmos, HUD text
        if np.abs(c[[0, 2]]).max() < 3 and (hi - lo).max() < 150: return True
        s = np.linalg.norm(p.W[:3, :3], axis=0)
        return s.max() / s.min() > 1.05
    keep, mirrored, origin = [], 0, 0
    for p in parts:
        # the reflection pass mirrors about y = 0, so its copies end up underground; a mirrored part above ground
        # is real (the far-side half of a pinch brake is the near-side mesh mirrored)
        if np.linalg.det(p.W[:3, :3]) < 0 and p.V[:, 1].max() < 0.5: mirrored += 1; continue
        if at_origin(p) or (p.stride == 20 and len(p.V) <= 4):    # lone triangles / quads: text glyphs, particles
            origin += 1; continue
        keep.append(p)
    log(f"dropped {mirrored} mirrored reflection-pass copies and {origin} unplaced meshes at the origin")
    return keep

def drop_shadow_copies(parts, log=print):
    """shadow-map passes draw scenery with no texture, and their recovered matrices can be up to ~1 m off the
    main view's, too far for the dedup in collect_parts: a plain white copy then sits beside the textured
    object (a white rail beside a wooden fence). Drop an untextured part when the same mesh, turned the same
    way, is also there textured close by (1 m for scenery; for the ride, only exact overlaps)."""
    untex = lambda p: p.M is None or p.M[0] is None
    by = collections.defaultdict(lambda: ([], []))
    for p in parts: by[p.mesh_key][untex(p)].append(p)
    drop = set()
    for tex, plain in by.values():
        if not tex or not plain: continue
        T = np.array([q.W[:3, 3] for q in tex]); R = np.array([q.W[:3, :3] for q in tex])
        for p in plain:
            near = np.abs(T - p.W[:3, 3]).max(1) < (1.0 if p.group == "scenery" else 0.05)
            if (near & (np.abs(R - p.W[:3, :3]).max((1, 2)) < 0.1)).any(): drop.add(id(p))
    if drop: log(f"dropped {len(drop)} untextured shadow-pass copies")
    return [p for p in parts if id(p) not in drop]

def drop_stray_instances(parts, log=print, reach=1.5):
    """bolts and brackets are instanced in bulk; some instance records were stale or belonged to supports that
    were never captured and come out as loose grids of bolts on the ground or hanging in the air. A small mesh
    whose instances mostly sit on the ride keeps only the instances that touch something."""
    inst = collections.defaultdict(list)
    for p in parts:
        if p.group in ("structure", "scenery") and not identity(p.W) and (p.V.max(0) - p.V.min(0)).max() < 1.5:
            inst[(p.mesh_key[0], len(p.V))].append(p)
    cand = {k for k, L in inst.items() if len(L) >= 20}
    if not cand: return parts
    small = {id(p) for k in cand for p in inst[k]}
    host = [p for p in parts if id(p) not in small and p.group in ("structure", "track", "scenery")]
    g = Grid(np.concatenate([p.V[:: max(1, len(p.V) // 4000)] for p in host]), cell=2.0)
    drop = set()
    for k in cand:
        L = inst[k]; d = np.array([g.dist(p.W[:3, 3]) for p in L])
        if (d < 1.0).mean() < 0.6: continue          # a scenery object in its own right (lamps, bins...)
        drop.update(id(p) for p, dd in zip(L, d) if dd > reach)
    if drop: log(f"dropped {len(drop)} loose bolts / brackets not attached to anything")
    return [p for p in parts if id(p) not in drop]

_FOLIAGE = {}
def is_foliage(tid):
    """leaf / needle / flower cut-out textures: mostly clear, and what is opaque is strongly coloured and green
    rather than blue (grating and fences are grey, signs are mostly opaque)"""
    if not tid or tid not in TEXTURES: return False
    if tid not in _FOLIAGE:
        img = TEXTURES[tid]; r = False
        if _alpha_mode_arr(img) == "MASK":
            op = img[..., 3] > 128; f = float(op.mean())
            if 0.02 < f < 0.4:
                rgb = img[..., :3][op].astype(np.float32)
                if len(rgb) > 20000: rgb = rgb[:: len(rgb) // 20000]
                mx, mn = rgb.max(1), rgb.min(1)
                sat = float(np.median((mx - mn) / np.maximum(mx, 1)))
                green = float(((rgb[:, 1] >= rgb[:, 2]) & (rgb[:, 0] < 1.4 * rgb[:, 1] + 10)).mean())
                r = sat >= 0.4 and green >= 0.6
        _FOLIAGE[tid] = r
    return _FOLIAGE[tid]

def _ground_tile(p):
    e = p.V.max(0) - p.V.min(0)
    if min(e[0], e[2]) >= 40 and e[1] < 1.0: return True
    # a 64 m terrain tile with a bit of height in it (the station plaza one rises 1.3 m)
    return bool(np.abs(e[[0, 2]] - 64).max() < 0.5 and e[1] < 3.0 and identity(p.W)
                and np.abs(p.V.min(0)[[0, 2]] / 64 - np.round(p.V.min(0)[[0, 2]] / 64)).max() < 0.01)

GRASS = (0.13, 0.20, 0.06)   # linear

def drop_back_faces(parts, log=print):
    """thin sheets (roofs, signs, leaves) are modelled as two faces on the same spot facing opposite ways; the game
    culls the one facing away, but the exported materials are double-sided, so Blender draws both and they z-fight
    (dark stripes on the transfer shed roof). Keep one face of each such pair; double-sided shows it from both sides."""
    n = 0
    for p in parts:
        F = p.F
        if len(F) < 2: continue
        q = np.round(p.V.astype(np.float64) * 1000).astype(np.int64)
        T = q[F]                                                   # (n, 3, 3)
        o = np.lexsort((T[:, :, 2], T[:, :, 1], T[:, :, 0]), axis=1)   # vertices of each triangle in a fixed order
        key = np.take_along_axis(T, o[:, :, None], 1).reshape(len(F), 9)
        _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        if cnt.max() < 2: continue
        V = p.V.astype(np.float64)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        keep = np.ones(len(F), bool)
        order = np.argsort(inv, kind="stable"); groups = np.split(order, np.cumsum(cnt)[:-1])
        for g in np.where(cnt >= 2)[0]:
            idx = groups[g]
            if (fn[idx] @ fn[idx[0]]).min() < 0: keep[idx[1:]] = False   # opposite-facing twins: keep the first
        if not keep.all(): n += int((~keep).sum()); p.F = np.ascontiguousarray(F[keep])
    if n: log(f"dropped {n} back-to-back twin faces (double-sided sheets)")

def fix_ground(parts, log=print):
    """NL2 draws the ground in 64 m tiles with a terrain shader that blends several textures; whatever is bound
    in slot 0 is not its colour (the capture shows bark, leaves...). Keep one tile per spot (the finest; other
    captures and LODs add overlapping copies) and give them a plain grass colour."""
    tiles = [p for p in parts if p.group == "terrain" and _ground_tile(p)]
    best = {}
    for p in tiles:
        k = tuple(np.round(p.V.min(0)[[0, 2]] / 4).astype(int)) + tuple(np.round(p.V.max(0)[[0, 2]] / 4).astype(int))
        q = best.get(k)
        if q is None or (identity(p.W), len(p.F)) > (identity(q.W), len(q.F)): best[k] = p
    keep = {id(p) for p in best.values()}
    for p in best.values(): p.M, p.C = (None, GRASS), None
    log(f"ground: {len(best)} tiles ({len(tiles) - len(best)} overlapping copies dropped)")
    return [p for p in parts if not (p.group == "terrain" and _ground_tile(p)) or id(p) in keep]

def remove_trees(parts, log=print):
    """drop trees, bushes and plants: every leaf-textured part, plus the trunk / branch meshes placed with the
    same world matrix inside the plant's footprint"""
    fol = [p for p in parts if p.M and is_foliage(p.M[0]) and not _ground_tile(p)]
    if not fol: return parts
    # the same leaf mesh drawn by a pass that never gave it a texture (shadow maps; distant forest card batches)
    shape = lambda p: (len(p.local), len(p.F)) + tuple(np.round(p.local.max(0) - p.local.min(0), 1))
    shapes = {shape(p) for p in fol if len(p.F) >= 8}
    fol += [p for p in parts if not (p.M and p.M[0]) and len(p.F) >= 8 and shape(p) in shapes]
    drop = {id(p) for p in fol}
    wk = lambda p: tuple(np.round(p.W[:3, :].ravel(), 2))
    box = {}
    for p in fol:
        if identity(p.W): continue
        k = wk(p); lo, hi = p.V.min(0), p.V.max(0)
        box[k] = (np.minimum(box[k][0], lo), np.maximum(box[k][1], hi)) if k in box else (lo, hi)
    for p in parts:
        if id(p) in drop or identity(p.W): continue
        b = box.get(wk(p))
        if b is None: continue
        lo, hi = p.V.min(0), p.V.max(0); e = hi - lo
        if e.max() < 0.3 or e.max() > 40: continue     # bolts that share the matrix; whole buildings
        m = 0.25 * (b[1] - b[0])[[0, 2]].max() + 0.5
        if lo[0] >= b[0][0] - m and hi[0] <= b[1][0] + m and lo[2] >= b[0][2] - m and hi[2] <= b[1][2] + m:
            drop.add(id(p))
    log(f"removed {len(drop)} tree / plant parts ({len(fol)} leaf meshes)")
    return [p for p in parts if id(p) not in drop]

class Grid:
    """nearest-point queries without scipy"""
    def __init__(self, P, cell=4.0):
        self.cell = cell; self.g = collections.defaultdict(list)
        for p in P: self.g[tuple((p // cell).astype(int))].append(p)
        self.g = {k: np.array(v) for k, v in self.g.items()}
    def dist(self, q):
        c = (q // self.cell).astype(int); best = 1e9
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    a = self.g.get((c[0] + dx, c[1] + dy, c[2] + dz))
                    if a is not None: best = min(best, float(np.min(np.linalg.norm(a - q, axis=1))))
        return best

def identity(W): return np.allclose(W[:3, :3], np.eye(3), atol=1e-3) and np.abs(W[:3, 3]).max() < 1e-3

def clean_track(parts, log=print, reach=0.75, need=0.9):
    """the shadow/depth passes redraw the track with coarser, unpainted meshes; drop any unpainted track mesh that
    lies on painted track (these came out white/grey and doubled up the rails, spine and crossties)"""
    tr = [p for p in parts if is_track(p)]
    painted = [p for p in tr if p.C is not None]
    if not painted: return parts
    g = Grid(np.concatenate([p.V for p in painted]), cell=reach)
    drop, kept = set(), []
    for p in tr:
        if p.C is not None: continue
        S = p.V[:: max(1, len(p.V) // 400)]
        cov = float(np.mean([g.dist(v) < reach for v in S]))
        if cov >= need: drop.add(id(p))
        else: kept.append((p, cov))
    for p, cov in kept:
        log(f"  unpainted track kept: {len(p.V)} verts, {cov:.0%} on painted track, {p.V.min(0).round(0)}..{p.V.max(0).round(0)}")
    # the painted track itself comes in several detail levels (one per capture distance) that pile up on top of each
    # other; per track component (vertex buffer + material) keep the finest mesh of each stretch
    comp = collections.defaultdict(list)
    for p in painted:
        cs = set(map(tuple, np.unique(np.floor(p.V).astype(np.int64), axis=0)))
        comp[(p.key[0], p.M[0] if p.M else None)].append((len(p.F) / len(cs), len(p.F), p, cs))
    lod = 0
    for L in comp.values():
        L.sort(key=lambda t: (-t[0], -t[1])); occ = set()
        for _, _, p, cs in L:
            if len(cs - occ) <= 0.1 * len(cs): drop.add(id(p)); lod += 1
            else: occ |= cs
    log(f"track: dropped {len(drop) - lod} unpainted shadow/LOD copies and {lod} overlapping coarser copies, "
        f"kept {len(painted) - lod} painted meshes")
    parts = [p for p in parts if id(p) not in drop]
    return parts

def remove_rides(parts, seeds, log=print, cell=1.0, seed_r=6.0):
    """drop other rides (another coaster, a tower...) standing in the scenery: flood-fill from a seed point (x, z)
    through touching scenery/structure pieces, never crossing into pieces within 4 m of the exported track"""
    if not seeds: return parts
    tr = [p for p in parts if p.group == "track"]
    g = Grid(np.concatenate([p.V[::2] for p in tr]), cell=4.0)
    cand = [p for p in parts if p.group in ("structure", "scenery") and (p.V.max(0) - p.V.min(0)).max() < 60]
    keys = [set(map(tuple, np.unique(np.floor(p.V[:: max(1, len(p.V) // 2000)] / cell).astype(np.int64), axis=0))) for p in cand]
    owner = collections.defaultdict(list)
    for i, ks in enumerate(keys):
        for k in ks: owner[k].append(i)
    blocked = [g.dist(p.V.mean(0)) < 4.0 for p in cand]
    ctr = np.array([p.V.mean(0)[[0, 2]] for p in cand])
    drop = set()
    for sx, sz in seeds:
        n0 = len(drop)
        st = [i for i in np.nonzero(np.linalg.norm(ctr - (sx, sz), axis=1) < seed_r)[0] if not blocked[i] and i not in drop]
        drop.update(st)
        while st:
            for k in keys[st.pop()]:
                for j in owner[k]:
                    if j not in drop and not blocked[j]: drop.add(j); st.append(j)
        log(f"rides: removed {len(drop) - n0} pieces around ({sx:g}, {sz:g})")
    ids = {id(cand[i]) for i in drop}
    # long pieces of those rides (a wooden coaster's rails, a tower's mast) were too big to flood through: drop
    # anything that lies over the removed rides' footprint and not near the track
    if drop:
        foot = set()
        for i in drop:
            for c in np.unique(np.floor(cand[i].V[:, [0, 2]] / 4.0).astype(np.int64), axis=0):
                foot.update((int(c[0]) + dx, int(c[1]) + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1))
        big = 0
        for p in parts:
            if id(p) in ids or p.group not in ("structure", "scenery", "terrain") or _ground_tile(p): continue
            S = p.V[:: max(1, len(p.V) // 300)]
            inside = np.mean([(int(a), int(b)) in foot for a, b in np.floor(S[:, [0, 2]] / 4.0).astype(np.int64)])
            if inside >= 0.8 and np.mean([g.dist(v) < 6.0 for v in S[:: max(1, len(S) // 40)]]) < 0.1:
                ids.add(id(p)); big += 1
        if big: log(f"rides: removed {big} more large pieces over those rides")
    return [p for p in parts if id(p) not in ids]

def is_track(p):
    """the engine's world-space track meshes: identity world matrix, 40-byte vertices; the finest detail level
    (always used by a full-scene capture) has 64-byte vertices. Painted with the 8-texel ramp -> per-vertex C."""
    return (identity(p.W) and p.stride == 40) or (p.C is not None and p.stride in (40, 64))

def classify(parts, log=print):
    track = [p for p in parts if is_track(p)]
    for p in track: p.group = "track"
    if not track:
        log("no track found (was the coaster on screen?)"); return None
    grid = Grid(np.concatenate([p.V[::4] for p in track]))
    # terrain: large flat world-space meshes
    for p in parts:
        if p.group: continue
        ext = p.V.max(0) - p.V.min(0)
        if identity(p.W) and p.stride == 24 and min(ext[0], ext[2]) > 20 and ext[1] < 0.5 * max(ext[0], ext[2]): p.group = "terrain"
        elif min(ext[0], ext[2]) > 15 and ext[1] < 0.5 and len(p.F) <= 64: p.group = "terrain"   # flat ground patches / paths
        elif _ground_tile(p): p.group = "terrain"   # 64 m ground tiles in other vertex formats
    # trains: every car carries the same set of parts, so car-part shapes repeat exactly n_cars times
    # (or a multiple, e.g. 4 seats per car) and sit on the track. Shape = vertex count + local size.
    def shape(p):
        e = p.local.max(0) - p.local.min(0)
        return (len(p.local),) + tuple(np.round(e, 2))
    reps = collections.defaultdict(list)
    for p in parts:
        if p.group or identity(p.W): continue
        # bolts/nuts are instances of a ~2 m mesh scaled down to a few cm (x0.02-0.07), repeated thousands of times
        # along the track: not car parts (train models are placed at x0.9-1.0)
        if np.linalg.norm(p.W[:3, :3], axis=0).max() < 0.5: continue
        if (p.local.max(0) - p.local.min(0)).max() < 4.5: reps[shape(p)].append(p)
    near = {k: float(np.median([grid.dist(p.W[:3, 3]) for p in L])) for k, L in reps.items()}
    # car count: most common repeat count among car-sized parts (1.2 m .. 3.5 m) on the track
    cc = collections.Counter(len(L) for k, L in reps.items() if near[k] < 3.0 and 1.2 <= max(k[1:]) <= 3.5 and len(L) >= 2)
    train_shapes = set()
    if cc:
        n_cars = max(cc, key=lambda n: cc[n] * n)   # per-car parts beat per-train ones (e.g. the nose, once per train)
        log(f"trains: {n_cars} cars on screen")
        for k, L in reps.items():
            if near[k] < 3.0 and len(L) % n_cars == 0 and len(L) <= 8 * n_cars: train_shapes.add(k)
    train = [p for p in parts if p.group is None and not identity(p.W) and shape(p) in train_shapes]
    for p in train: p.group = "trains"; p.mesh_key = shape(p)
    if train:   # one-off pieces bolted to a train (front-car nose etc.)
        tp = Grid(np.array([p.W[:3, 3] for p in train]), cell=3.0)
        for p in parts:
            if p.group or identity(p.W) or np.linalg.norm(p.W[:3, :3], axis=0).max() < 0.5: continue
            ext = p.local.max(0) - p.local.min(0)
            if ext.max() < 4.5 and tp.dist(p.W[:3, 3]) < 2.0 and grid.dist(p.W[:3, 3]) < 3.0:
                p.group = "trains"; p.mesh_key = shape(p)
        filter_train_parts([p for p in parts if p.group == "trains"], log)
        adopt_car_parts(parts, log)
    # structure: anything else within reach of the track; scenery: the rest
    for p in parts:
        if p.group: continue
        ctr = p.V.mean(0); r = float(np.linalg.norm(p.V.max(0) - p.V.min(0))) / 2
        near = grid.dist(ctr) < 6.0 + r
        if not near and r > 10:   # a big mesh (a whole run of catwalk / lift chain) whose centre is off the track
            S = p.V[:: max(1, len(p.V) // 200)]
            near = np.mean([grid.dist(v) < 4.0 for v in S]) >= 0.5
        p.group = "structure" if near else "scenery"
    grow_structure(parts, log)
    return track

def grow_structure(parts, log=print, cell=0.4, max_steps=200):
    """the distance test above misses the low end of tall supports: bolts and flanges near the ground, the
    lower flights of stairs. Pull in small pieces (and thin poles) that touch the structure, and pieces that
    touch those, so chains of stair steps follow their tower down."""
    def keys(V):
        c = np.floor(V / cell).astype(np.int64) + (1 << 20)
        return np.unique((c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2])
    offs = np.array([(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)
    cand = []
    for p in parts:
        if p.group != "scenery": continue
        e = p.V.max(0) - p.V.min(0)
        if e.max() < 5 or (max(e[0], e[2]) < 1.5 and e[1] < 40): cand.append(p)
    if not cand: return
    ck = [keys(p.V[:: max(1, len(p.V) // 256)]) for p in cand]
    K = np.concatenate(ck); owner = np.repeat(np.arange(len(cand)), [len(k) for k in ck])
    # seeds: candidates within a cell of the structure
    occ = np.unique(np.concatenate([keys(p.V) for p in parts if p.group in ("structure", "track")]))
    seed = set()
    for dk in offs: seed.update(np.unique(owner[np.isin(K + dk, occ)]).tolist())
    # candidate neighbours: share a cell (or touch one)
    o = np.argsort(K, kind="stable"); Ks, Os = K[o], owner[o]
    adj = collections.defaultdict(set)
    for dk in offs:
        q = K + dk; lo = np.searchsorted(Ks, q, "left"); hi = np.searchsorted(Ks, q, "right")
        m = (hi > lo) & (hi - lo <= 64)          # a crowded cell (a pile of tiny bits): don't chain through it
        for a, l, h in zip(owner[m].tolist(), lo[m].tolist(), hi[m].tolist()):
            for b in Os[l:h].tolist():
                if b != a: adj[a].add(b)
    got = set(seed); front = list(got)
    for _ in range(max_steps):
        nxt = {b for a in front for b in adj.get(a, ()) if b not in got}
        if not nxt: break
        got |= nxt; front = list(nxt)
    for i in got: cand[i].group = "structure"
    if got: log(f"structure: {len(got)} pieces touching supports / stairs moved from scenery ({len(seed)} touching directly)")

# ------------------------------------------------------------------------------------------ train cars
def car_frames(train):
    """world frames of the cars: the most-repeated car-sized piece (chassis) of each car"""
    by = collections.defaultdict(list)
    for p in train: by[p.mesh_key].append(p)
    counts = collections.Counter(len(L) for k, L in by.items() if 1.2 <= max(k[1:]) <= 3.5)
    if not counts: return []
    n_cars = max(counts, key=lambda n: counts[n] * n)
    chassis_key = max((k for k, L in by.items() if len(L) == n_cars), key=lambda k: len(by[k][0].local))
    return [p.W for p in by[chassis_key]]

def split_trains(C):
    """group car centres into trains (chains of neighbouring cars), each ordered from one end"""
    C = np.asarray(C)
    if len(C) < 2: return [list(range(len(C)))]
    D = np.linalg.norm(C[:, None] - C[None], axis=2); np.fill_diagonal(D, 1e9)
    link = 1.6 * float(np.median(D.min(1)))
    trains_, seen_ = [], set()
    for s0 in range(len(C)):
        if s0 in seen_: continue
        comp, stack = [], [s0]; seen_.add(s0)
        while stack:
            k = stack.pop(); comp.append(k)
            for j in np.nonzero(D[k] < link)[0]:
                if int(j) not in seen_: seen_.add(int(j)); stack.append(int(j))
        start = max(comp, key=lambda k: np.linalg.norm(C[k] - C[comp].mean(0)))
        order_, left = [start], set(comp) - {start}
        while left:
            nxt = min(left, key=lambda j: D[order_[-1], j]); order_.append(nxt); left.discard(nxt)
        trains_.append(order_)
    return trains_

def filter_train_parts(train, log=print):
    """keep only pieces that ride with the cars. A real car piece sits at the same place relative to its car on
    every car (seats, wheels) or on the end car of every train (nose); brake fins, drive tyres and other track
    parts under a parked train only line up with a car by chance. Rejected pieces go back to the layout."""
    frames = car_frames(train)
    if len(frames) < 2: return
    C = np.array([F[:3, 3] for F in frames]); trains_ = split_trains(C)
    tid = {k: t for t, L in enumerate(trains_) for k in L}
    ends = {k for t in trains_ for k in (t[0], t[-1])}
    inv = [np.linalg.inv(F) for F in frames]
    rec = []
    for p in train:
        d = np.linalg.norm(C - p.V.mean(0), axis=1); k = int(np.argmin(d))
        rec.append((p, k, inv[k] @ p.W) if d[k] < 3.5 else (p, None, None))
    cell = 0.1; H = collections.defaultdict(list)
    for n, (p, k, T) in enumerate(rec):
        if k is not None: H[tuple(np.floor(T[:3, 3] / cell).astype(int))].append(n)
    n_cars, n_tr, dropped = len(frames), len(trains_), 0
    for p, k, T in rec:
        ok = False
        if k is not None:
            c = np.floor(T[:3, 3] / cell).astype(int)
            cars = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for m in H.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                            p2, k2, T2 = rec[m]
                            # same place on the car; bogies may swivel a little on curves
                            if np.abs(T2[:3, 3] - T[:3, 3]).max() < 0.08 and np.abs(T2[:3, :3] - T[:3, :3]).max() < 0.1:
                                cars.add(k2)
            trs = {tid[q] for q in cars}
            if len(cars) >= max(2, 0.6 * n_cars): ok = True                    # on (nearly) every car
            elif n_tr >= 2 and k in ends and cars <= ends and len(trs) >= max(2, 0.6 * n_tr): ok = True   # nose / tail
            elif n_tr == 1 and (len(cars) >= 2 or k in ends): ok = True        # one train: can't cross-check the ends
        if not ok: p.group = None; dropped += 1
    if dropped: log(f"trains: {dropped} track pieces near the trains (brakes, tyres...) moved back to the layout")

def adopt_car_parts(parts, log=print):
    """wheels spin and bogies swivel, so their copies don't repeat with the exact same matrix and the car-part test
    misses them; take any small piece that sits at the same place on (nearly) every car, rotation aside"""
    train = [p for p in parts if p.group == "trains"]
    frames = car_frames(train)
    if len(frames) < 2: return
    C = np.array([F[:3, 3] for F in frames]); inv = [np.linalg.inv(F) for F in frames]
    def shape(p):
        e = p.local.max(0) - p.local.min(0); return (len(p.local),) + tuple(np.round(e, 2))
    at = collections.defaultdict(list)
    for p in parts:
        if p.group == "trains" or identity(p.W) or (p.local.max(0) - p.local.min(0)).max() > 1.5: continue
        d = np.linalg.norm(C - p.W[:3, 3], axis=1); k = int(np.argmin(d))
        if d[k] < 3.0: at[shape(p)].append((k, (inv[k] @ p.W)[:3, 3], p))
    n = 0
    for s, L in at.items():
        pos = np.array([t for _, t, _ in L])
        for q in np.unique(np.round(pos / 0.05).astype(int), axis=0):   # each place on the car this shape sits
            sel = [i for i in range(len(L)) if np.abs(pos[i] - q * 0.05).max() < 0.08]
            if len({L[i][0] for i in sel}) >= 0.6 * len(frames):
                for i in sel:
                    if L[i][2].group != "trains": L[i][2].group = "trains"; L[i][2].mesh_key = s; n += 1
    if n: log(f"trains: {n} wheels / bogie pieces found on every car added back")
    # the lead car runs on a second bogie one car-pitch ahead: a copy of the car's own bogie on its free side.
    # Only 1 car per train has it, so the pass above misses its spinning wheels
    train = [p for p in parts if p.group == "trains"]
    near = collections.defaultdict(list)
    for p in train:
        d = np.linalg.norm(C - p.W[:3, 3], axis=1); k = int(np.argmin(d))
        if d[k] < 3.0: near[k].append(p)
    D = np.linalg.norm(C[:, None] - C[None], axis=2); np.fill_diagonal(D, 1e9)
    pitch = float(np.median(D.min(1))); m = 0
    for k in range(len(C)):
        nb = np.nonzero(D[k] < 1.6 * pitch)[0]
        if len(nb) != 1: continue
        off = -(inv[k][:3, :3] @ (C[nb[0]] - C[k]))           # car-local step to the free side, one pitch
        own = collections.defaultdict(list)
        for p in near[k]:
            t = (inv[k] @ p.W)[:3, 3]
            if np.abs(t).max() < 1.0: own[shape(p)].append(t + off)
        cand = [p for p in parts if p.group != "trains" and not identity(p.W) and shape(p) in own
                and np.linalg.norm(p.W[:3, 3] - C[k]) < 3.0]
        hit = [p for p in cand if any(np.abs((inv[k] @ p.W)[:3, 3] - q).max() < 0.08 for q in own[shape(p)])]
        if len(hit) >= 6:
            for p in hit: p.group = "trains"; p.mesh_key = shape(p); m += 1
    if m: log(f"trains: {m} lead-bogie wheels added to the front cars")

# colours the capture doesn't carry (white texture, factor 1): the orange bumpers on the train ends (linear RGB)
TRAIN_TINT = {"e5fe316984a3": (0.52, 0.09, 0.025)}

def extract_cars(parts, log=print, all_cars=False):
    train = [p for p in parts if p.group == "trains"]
    if not train: return []
    frames = car_frames(train)
    if not frames: return []
    centres = [F[:3, 3] for F in frames]
    cars = [[] for _ in frames]
    for p in train:
        if len(p.local) <= 36 and len(p.F) <= 12 and (p.M is None or p.M[0] is None): continue   # plain marker cubes
        c = p.V.mean(0)
        d = [np.linalg.norm(c - q) for q in centres]
        k = int(np.argmin(d))
        if d[k] < 3.5: cars[k].append(p)          # drops chain links / catwalk bits that were caught nearby
    inv = [np.linalg.inv(F) for F in frames]
    def to_local(k, V): return V @ inv[k][:3, :3].T + inv[k][:3, 3]
    # car order along the train, and the local axes: "forward" = towards the neighbouring car, "up" = world up
    rest_ = set(range(len(frames)))
    # split the cars into trains (chains of neighbouring cars), then order each chain from one end
    C = np.array(centres); D = np.linalg.norm(C[:, None] - C[None], axis=2); np.fill_diagonal(D, 1e9)
    link = 1.6 * float(np.median(D.min(1))) if len(C) > 1 else 0.0
    trains_, seen_ = [], set()
    for s0 in range(len(C)):
        if s0 in seen_: continue
        comp, stack = [], [s0]; seen_.add(s0)
        while stack:
            k = stack.pop(); comp.append(k)
            for j in np.nonzero(D[k] < link)[0]:
                if int(j) not in seen_: seen_.add(int(j)); stack.append(int(j))
        start = max(comp, key=lambda k: np.linalg.norm(C[k] - C[comp].mean(0)))
        order_, left = [start], set(comp) - {start}
        while left:
            nxt = min(left, key=lambda j: D[order_[-1], j]); order_.append(nxt); left.discard(nxt)
        trains_.append(order_)
    order = [k for t in trains_ for k in t]
    fwd = np.zeros(3); up = np.zeros(3)
    for a_, b_ in [(t[i], t[i + 1]) for t in trains_ for i in range(len(t) - 1)]:
        fwd += inv[a_][:3, :3] @ (centres[b_] - centres[a_])
        up += inv[a_][:3, :3] @ np.array([0.0, 1.0, 0.0])
    ax_f = int(np.argmax(np.abs(fwd))); ax_u = max((i for i in range(3) if i != ax_f), key=lambda i: abs(up[i]))
    # pieces hanging between two cars (the lap-bar covers reach under the seat backs of the car ahead) sit closer to
    # the neighbour's frame than to their own, so they get handed to the wrong car. The end car gives it away: it
    # carries such a piece twice, at P and again one car further along (P + d, the neighbour's copy). Hand every
    # piece found at P + d back to that neighbour, all along the train.
    def lpos(k, p): return (inv[k] @ p.W)[:3, 3]
    moved = 0
    for t in trains_:
        if len(t) < 2: continue
        ends = []
        for e, n, step in ((t[0], t[1], 1), (t[-1], t[-2], -1)):
            d = inv[e][:3, :3] @ (centres[n] - centres[e])
            here = collections.defaultdict(list)
            for p in cars[e]: here[p.mesh_key].append(lpos(e, p))
            ends.append((e, step, [(s, q) for s, P in here.items() for q in P
                                   if any(np.abs(q - (r + d)).max() < 0.08 for r in P)]))
        # bogies between cars look doubled at both ends; only one-per-seat shapes doubled at a single end are strays
        cnt = [collections.Counter(s for s, q in x) for _, _, x in ends]
        # a shape doubled on one end car is only a stray when the car at the other end is missing it (the covers);
        # if that car has it too, the double is real (the lead car's second bogie)
        for j, (e, step, extra) in enumerate(ends):
            e2 = ends[1 - j][0]
            here2 = collections.defaultdict(list)
            for p in cars[e2]: here2[p.mesh_key].append(lpos(e2, p))
            extra = [(s, q) for s, q in extra if cnt[j][s] >= 4
                     and not any(np.abs(r - q).max() < 0.08 for r in here2.get(s, ()))]
            if not extra: continue
            for i in (range(len(t) - 1) if step == 1 else range(len(t) - 1, 0, -1)):
                k, nb = t[i], t[i + step]
                give = [p for p in cars[k] if any(p.mesh_key == s and np.abs(lpos(k, p) - q).max() < 0.08 for s, q in extra)]
                if give:
                    cars[k] = [p for p in cars[k] if not any(p is g for g in give)]; cars[nb] += give; moved += len(give)
    if moved: log(f"trains: {moved} pieces between cars (lap-bar covers...) given back to their own car")
    # parts every car has = the car body; a piece found in one car only is either real (nose, higher-LOD seat)
    # or a stray neighbour; keep it only if it sits inside the car's body outline (the nose may stick out forward)
    common = set.intersection(*[set(p.mesh_key for p in L) for L in cars if L]) if any(cars) else set()
    count = collections.Counter(p.mesh_key for L in cars for p in L)
    protrude = [0.0] * len(cars)
    # a piece found at the same car-local spot on 3+ cars (bumpers on every train's ends) is real, even if it sticks out
    seen_at = collections.defaultdict(list)
    for k, L in enumerate(cars):
        for p in L:
            if p.mesh_key not in common: seen_at[p.mesh_key].append((k, lpos(k, p)))
    endn = {t[0]: t[1] for t in trains_ if len(t) > 1}; endn.update({t[-1]: t[-2] for t in trains_ if len(t) > 1})
    def recurs(k, p):
        if k not in endn: return False
        q = lpos(k, p)
        if float(q @ (inv[k][:3, :3] @ (centres[endn[k]] - centres[k]))) > 0: return False   # neighbour side = stray
        return len({j for j, r in seen_at[p.mesh_key] if j != k and np.abs(r - q).max() < 0.08}) >= 2
    for k, L in enumerate(cars):
        core = [to_local(k, p.V) for p in L if p.mesh_key in common]
        if not core: continue
        allc = np.concatenate(core); mn, mx = allc.min(0), allc.max(0)
        keep = []
        for p in L:
            if p.mesh_key in common: keep.append(p); continue
            Vl = to_local(k, p.V); lo, hi = mn - 0.25, mx + 0.25
            lo2, hi2 = lo.copy(), hi.copy(); lo2[ax_f] -= 1.5; hi2[ax_f] += 1.5
            if np.all(Vl.min(0) >= lo2) and np.all(Vl.max(0) <= hi2) and (len(p.F) >= 300 or recurs(k, p) or (np.all(Vl.min(0) >= lo) and np.all(Vl.max(0) <= hi))):
                keep.append(p)
                protrude[k] = max(protrude[k], float(max(Vl[:, ax_f].max() - mx[ax_f], mn[ax_f] - Vl[:, ax_f].min(), 0)))
        cars[k] = keep
    # a car can hold the same piece at two detail levels (e.g. shadow pass uses a coarser mesh): keep the finest
    for k, L in enumerate(cars):
        info = [(p, p.V.mean(0), p.V.max(0) - p.V.min(0)) for p in sorted(L, key=lambda q: -len(q.F))]
        keep = []
        for p, c_, e_ in info:
            if any(p.mesh_key == q.mesh_key and np.abs(c_ - c2).max() < 0.03 for q, c2, e2 in keep): continue   # same wheel seen twice
            if any(np.abs(c_ - c2).max() < 0.02 + 0.03 * e2.max() and np.abs(e_ - e2).max() < 0.03 * e2.max() + 0.01
                   and p.mesh_key != q.mesh_key and len(p.F) < 0.7 * len(q.F) for q, c2, e2 in keep): continue
            keep.append((p, c_, e_))
        cars[k] = [p for p, _, _ in keep]
    tris = [sum(len(p.F) for p in L) for L in cars]
    end_cars = [k for t in trains_ for k in (t[0], t[-1])]
    front = max(end_cars, key=lambda k: (protrude[k], tris[k]))   # the nose sticks out ahead of the car body
    # the other end cars carry the rear bumper: a middle car is one with a neighbour on both sides
    ends = set(end_cars)
    rear_of = {t[0]: t[-1] for t in trains_ if len(t) > 1}; rear_of.update({t[-1]: t[0] for t in trains_ if len(t) > 1})
    rears = [k for k in ends if k != front and cars[k] and protrude[k] < 0.5 * max(protrude[front], 0.2)]
    rear = rear_of.get(front) if rear_of.get(front) is not None and cars[rear_of[front]] else (max(rears, key=lambda k: tris[k]) if rears else None)
    rest = [k for k in range(len(cars)) if cars[k] and k not in ends] or \
           [k for k in range(len(cars)) if cars[k] and protrude[k] < 0.5 * max(protrude[front], 0.2)] or \
           [k for k in range(len(cars)) if k != front and cars[k]]
    types = collections.OrderedDict()
    types["front"] = (frames[front], cars[front])
    if rest:
        mid = max(rest, key=lambda k: tris[k])        # most detailed regular car (closest to the camera)
        types["car"] = (frames[mid], cars[mid])
    if rear is not None:
        types["rear"] = (frames[rear], cars[rear])
    if all_cars:
        for n_, k in enumerate(order):
            if cars[k]: types[f"car{n_ + 1}"] = (frames[k], cars[k])
    # rotation from the chassis frame to output axes: forward = +X, up = +Y (sign of forward: nose points +X)
    ax_s = 3 - ax_f - ax_u
    Rout = np.zeros((3, 3)); Rout[0, ax_f] = 1; Rout[1, ax_u] = 1 if up[ax_u] >= 0 else -1
    nose = [to_local(front, p.V)[:, ax_f].mean() for p in cars[front] if p.mesh_key not in common and count[p.mesh_key] == 1]
    if nose and np.mean(nose) < 0: Rout[0, ax_f] = -1
    Rout[2] = np.cross(Rout[0], Rout[1])
    out, shift = [], None
    for name, (F, L) in sorted(types.items(), key=lambda t: t[0] != "car"):
        iF = np.linalg.inv(F)
        car = []
        for p in L:
            T = iF @ p.W; T[:3] = Rout @ T[:3]; R = T[:3, :3]
            V = (p.local @ R.T + T[:3, 3]).astype(np.float32)
            N = (p.localN @ (R / np.linalg.norm(R, axis=0, keepdims=True)).T).astype(np.float32) if p.localN is not None else None
            M = (p.M[0], TRAIN_TINT[p.M[0]]) if p.M and p.M[0] in TRAIN_TINT else p.M
            car.append((V, N, p.UV, p.F, p.C, M))
        if shift is None:   # every car type shares the middle car's origin (centre of the car, floor at y=0), so
            allV = np.concatenate([c[0] for c in car]); mn, mx = allV.min(0), allV.max(0)   # they line up when
            shift = np.array([(mn[0] + mx[0]) / 2, mn[1], (mn[2] + mx[2]) / 2], np.float32)  # spaced one pitch apart
        out.append((name, [(V - shift, N, UV, F, C, M) for V, N, UV, F, C, M in car]))
    out.sort(key=lambda t: list(types).index(t[0]))
    pitch = [float(np.linalg.norm(centres[t[i + 1]] - centres[t[i]])) for t in trains_ for i in range(len(t) - 1)]
    log(f"trains: {len(frames)} cars in {len(trains_)} train(s)" + (f", car spacing {np.median(pitch):.3f} m" if pitch else ""))
    return out

# ------------------------------------------------------------------------------------------ writers
COLORS = {"track": (0.20, 0.40, 0.85), "structure": (0.55, 0.57, 0.60), "trains": (0.85, 0.15, 0.15),
          "scenery": (0.35, 0.55, 0.30), "terrain": (0.45, 0.40, 0.30)}

def _mat_name(gname, M):
    if M is None: return gname
    tid, f = M
    return f"{gname}_" + (f"t{tid}_" if tid else "") + "%02x%02x%02x" % tuple(int(round(v * 255)) for v in f)

def _alpha_mode(tid): return _alpha_mode_arr(TEXTURES[tid])

def _alpha_mode_arr(img):
    """cut-outs (leaves, fences, grating) have clearly opaque and clearly clear areas; other alpha channels in
    NL2 textures are masks (rust, gloss) and must not become transparency"""
    a = img[..., 3]
    if a.min() == 255: return None
    return "MASK" if (a > 240).mean() > 0.05 and (a < 16).mean() > 0.02 else None

def _item(it):   # (V, N, UV, F[, C, M]) -> 6-tuple
    return tuple(it) + (None,) * (6 - len(it))

def write_obj(path, groups, mtl=None):
    """OBJ + MTL; textures go to <dir>/textures/<id>.png. Per-vertex paint colours (track palette) are
    written as OBJ vertex colours (v x y z r g b), which Blender imports as a colour attribute."""
    import nl2_textures
    used = collections.OrderedDict()
    with open(path, "w", newline="\n") as f:
        f.write("# exported by NL2ModelExport\n")
        if mtl: f.write(f"mtllib {os.path.basename(mtl)}\n")
        vb = 1
        for gname, items in groups:
            f.write(f"g {gname}\n")
            cur = None
            for k, it in enumerate(items):
                V, N, UV, F, C, M = _item(it)
                mn = _mat_name(gname, M); used[mn] = (gname, M)
                f.write(f"o {gname}_{k:05d}\n")
                if mtl and mn != cur: f.write(f"usemtl {mn}\n"); cur = mn
                if C is not None and len(C) == len(V):
                    f.write("".join(f"v {a:.4f} {b:.4f} {c:.4f} {r:.3f} {g:.3f} {bb:.3f}\n" for (a, b, c), (r, g, bb) in zip(V, C)))
                else:
                    f.write("".join(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in V))
                hasN, hasT = N is not None and len(N) == len(V), UV is not None and len(UV) == len(V)
                if hasT: f.write("".join(f"vt {a:.5f} {1 - b:.5f}\n" for a, b in UV))
                if hasN: f.write("".join(f"vn {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in N))
                for a, b, c in F + vb:
                    if hasT and hasN: f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                    elif hasN: f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
                    elif hasT: f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
                    else: f.write(f"f {a} {b} {c}\n")
                vb += len(V)
    if mtl:
        tdir = os.path.join(os.path.dirname(mtl), "textures")
        with open(mtl, "w", newline="\n") as f:
            for mn, (gname, M) in used.items():
                if M is None: r, g, b = COLORS.get(gname, (0.7, 0.7, 0.7)); tid = None
                else: (tid, (r, g, b)) = M
                f.write(f"newmtl {mn}\nKd {r} {g} {b}\nKa 0 0 0\nd 1\n")
                if tid and tid in TEXTURES:
                    os.makedirs(tdir, exist_ok=True); png = os.path.join(tdir, tid + ".png")
                    if not os.path.exists(png): open(png, "wb").write(nl2_textures.png_bytes(TEXTURES[tid]))
                    f.write(f"map_Kd textures/{tid}.png\n")
                    if _alpha_mode(tid): f.write(f"map_d textures/{tid}.png\n")
                f.write("\n")

def _vertex_normals(V, F):
    n = np.zeros_like(V, dtype=np.float32)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): np.add.at(n, F[:, k], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l == 0] = 1
    return (n / l).astype(np.float32)

def write_glb(path, groups):
    """binary glTF 2.0: one node per group, one primitive per material (geometry merged), textures embedded as PNG,
    track paint colours as COLOR_0"""
    import json as _json, nl2_textures
    bin_ = bytearray(); views, accs, meshes, nodes, mats, images, texs = [], [], [], [], [], [], []
    def add_view(data, target=None):
        off = len(bin_); bin_.extend(data)
        while len(bin_) % 4: bin_.append(0)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target: v["target"] = target
        views.append(v); return len(views) - 1
    def add(arr, target, comp, typ, minmax=False):
        a = np.ascontiguousarray(arr)
        acc = {"bufferView": add_view(a.tobytes(), target), "componentType": comp, "count": int(a.shape[0]), "type": typ}
        if minmax: acc["min"] = a.min(0).tolist(); acc["max"] = a.max(0).tolist()
        accs.append(acc); return len(accs) - 1
    tex_index = {}
    def texture(tid):
        if tid not in tex_index:
            images.append({"bufferView": add_view(nl2_textures.png_bytes(TEXTURES[tid])), "mimeType": "image/png", "name": tid})
            texs.append({"source": len(images) - 1}); tex_index[tid] = len(texs) - 1
        return tex_index[tid]
    mat_index = {}
    def material(gname, M):
        key = _mat_name(gname, M)
        if key not in mat_index:
            if M is None: tid, col = None, COLORS.get(gname, (0.7, 0.7, 0.7))
            else: tid, col = M
            m = {"name": key, "pbrMetallicRoughness": {"baseColorFactor": [float(c) for c in col] + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.8},
                 "doubleSided": True}
            if tid and tid in TEXTURES:
                m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": texture(tid)}
                am = _alpha_mode(tid)
                if am: m["alphaMode"] = am
                if am == "MASK": m["alphaCutoff"] = 0.5
            mats.append(m); mat_index[key] = len(mats) - 1
        return mat_index[key]
    for gname, items in groups:
        if not items: continue
        by = collections.OrderedDict()
        for it in items:
            it = _item(it); by.setdefault((_mat_name(gname, it[5]), it[4] is not None), []).append(it)
        prims = []
        for (mn, hasC), its in by.items():
            Vs, Ns, Ts, Cs, Is, base = [], [], [], [], [], 0
            for V, N, UV, F, C, M in its:
                V = np.asarray(V, np.float32); F = np.asarray(F, np.int64)
                Vs.append(V)
                Ns.append(np.asarray(N, np.float32) if N is not None and len(N) == len(V) and np.isfinite(N).all() else _vertex_normals(V, F))
                Ts.append(np.asarray(UV, np.float32) if UV is not None and len(UV) == len(V) and np.isfinite(UV).all() else np.zeros((len(V), 2), np.float32))
                if hasC: Cs.append(np.asarray(C, np.float32))
                Is.append((F + base).astype(np.uint32).ravel()); base += len(V)
            V = np.concatenate(Vs); N = np.concatenate(Ns); T = np.concatenate(Ts); I = np.concatenate(Is)
            l = np.linalg.norm(N, axis=1, keepdims=True); bad = l[:, 0] < 1e-6; l[bad] = 1; N = N / l; N[bad] = (0, 1, 0)
            attrs = {"POSITION": add(V, 34962, 5126, "VEC3", True), "NORMAL": add(N.astype(np.float32), 34962, 5126, "VEC3"),
                     "TEXCOORD_0": add(T, 34962, 5126, "VEC2")}
            if hasC: attrs["COLOR_0"] = add(np.concatenate(Cs), 34962, 5126, "VEC3")
            prims.append({"attributes": attrs, "indices": add(I, 34963, 5125, "SCALAR"), "material": material(gname, its[0][5])})
        meshes.append({"name": gname, "primitives": prims}); nodes.append({"name": gname, "mesh": len(meshes) - 1})
    doc = {"asset": {"version": "2.0", "generator": "NL2ModelExport"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
           "nodes": nodes, "meshes": meshes, "materials": mats, "accessors": accs, "bufferViews": views,
           "buffers": [{"byteLength": len(bin_)}]}
    if images: doc["images"] = images; doc["textures"] = texs
    js = _json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(bin_), 0x004E4942)); f.write(bin_)

def write_model(out, name, groups, fmt, log):
    if fmt in ("obj", "both"): write_obj(os.path.join(out, name + ".obj"), groups, os.path.join(out, name + ".mtl"))
    if fmt in ("glb", "both"): write_glb(os.path.join(out, name + ".glb"), groups)

# ------------------------------------------------------------------------------------------ main
def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-scenery", action="store_true")
    ap.add_argument("--with-terrain", action="store_true", help=argparse.SUPPRESS)   # terrain is now on by default
    ap.add_argument("--no-terrain", action="store_true", help="leave the ground meshes out of scenery")
    ap.add_argument("--all-cars", action="store_true", help="also write every car on its own")
    ap.add_argument("--keep-trees", action="store_true", help="keep trees, bushes and plants (left out by default)")
    ap.add_argument("--drop-ride", action="append", default=[], metavar="X,Z",
                    help="remove another ride standing at world position X,Z (everything touching it, except near the track); repeatable")
    ap.add_argument("--no-cache", action="store_true", help="ignore <out>/parts.cache and re-read the captures")
    ap.add_argument("--instances", metavar="PKL",
                    help="bolts/nuts/connectors harvested park-wide by harvest_instances.py; replaces the captures' own instanced parts")
    ap.add_argument("--dump", help=argparse.SUPPRESS)   # debug: pickle the final classified parts here
    ap.add_argument("--format", choices=["glb", "obj", "both"], default="glb", help="model file format (default glb)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = []
    def log(s): print(s); report.append(s)
    import pickle
    cache = os.path.join(a.out, "parts.cache")
    sig = sorted(os.path.abspath(p) for p in a.captures)
    if a.instances: sig.append((os.path.abspath(a.instances), os.path.getmtime(a.instances)))
    parts = None
    if os.path.exists(cache) and not a.no_cache:   # re-running on the same captures skips the slow part
        try:
            got = pickle.load(open(cache, "rb"))
            got_sig, parts = got[0], got[1]
            if len(got) > 2: TEXTURES.update(got[2])
            if got_sig != sig: parts = None
            else: log(f"reusing {len(parts)} placed parts from {cache}")
        except Exception: parts = None
    if parts is None:
        caps = [Capture(p) for p in a.captures]
        if a.instances:
            h = pickle.load(open(a.instances, "rb")); left = dict(h["inst"])
            for c in caps: c.skip_inst = True
            first = None
            for src in h["sources"]:
                c = Capture(src)
                if first is None: first = c
                else:   # instanced-only captures: judge overlay passes by the whole frame of the same session
                    c._generic = first.generic_textures(); c._gshaders = first.generic_shaders()
                    c._painted = first.painted_textures(); first.used_slots(""); c._used = first._used
                have = {(x["vb"][0], x["ib"], x["startIndex"], x["count"], x["baseVertex"]) for x in c.d}
                c.harvest = {k: left.pop(k) for k in list(left) if k in have}
                caps.append(c)
            log(f"{a.instances}: {sum(len(v) for v in h['inst'].values())} harvested instances of {len(h['inst'])} meshes"
                + (f" ({len(left)} meshes without a source capture)" if left else ""))
        parts = collect_parts(caps, log=log)
        try: pickle.dump((sig, parts, TEXTURES), open(cache, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e: log(f"(could not write cache: {e})")
    for p in parts: p.group = None
    parts = drop_bad_placements(parts, log)
    if not a.keep_trees: parts = remove_trees(parts, log)
    parts = clean_track(parts, log)
    log(f"{len(parts)} unique placed parts")
    if classify(parts, log=log) is None: return 1
    parts = remove_rides(parts, [tuple(float(v) for v in s.split(",")) for s in a.drop_ride], log)
    parts = drop_stray_instances(parts, log)
    parts = drop_shadow_copies(parts, log)
    parts = fix_ground(parts, log)
    drop_back_faces(parts, log)
    cnt = collections.Counter(p.group for p in parts)
    log("groups: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
    if a.dump: pickle.dump((parts, TEXTURES), open(a.dump, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    order = ["track", "structure", "trains"]
    groups = [(g, [(p.V, p.N, p.UV, p.F, p.C, p.M) for p in parts if p.group == g]) for g in order]
    write_model(a.out, "layout", groups, a.format, log)
    log(f"wrote layout ({a.format}) ({sum(len(p.F) for p in parts if p.group in order)} triangles)")
    if any(p.group == "trains" for p in parts):
        write_model(a.out, "layout_no_trains", groups[:2], a.format, log)
        log(f"wrote layout_no_trains ({a.format}) ({sum(len(p.F) for p in parts if p.group in order[:2])} triangles)")
    extra = ([] if a.no_scenery else ["scenery"]) + ([] if a.no_terrain else ["terrain"])
    if extra:   # trees, buildings and ground go in their own file: they can be many times the size of the ride
        groups = [(g, [(p.V, p.N, p.UV, p.F, p.C, p.M) for p in parts if p.group == g]) for g in extra]
        write_model(a.out, "scenery", groups, a.format, log)
        log(f"wrote scenery ({a.format}) ({sum(len(p.F) for p in parts if p.group in extra)} triangles)")
    for name, car in extract_cars(parts, log=log, all_cars=a.all_cars):
        fn = {"front": "train_front_car", "car": "train_car", "rear": "train_rear_car"}.get(name, f"train_{name}")
        write_model(a.out, fn, [("trains", car)], a.format, log)
        log(f"wrote {fn} ({a.format}) ({sum(len(it[3]) for it in car)} triangles, {len(car)} pieces)")
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(report) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
