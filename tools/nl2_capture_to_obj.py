#!/usr/bin/env python3
"""Convert an NL2ModelExport capture folder into OBJ (and print analysis).

    py nl2_capture_to_obj.py <capture_dir> --analyze          per-draw table (to learn what is what)
    py nl2_capture_to_obj.py <capture_dir> -o out.obj [--min-verts N] [--draws 3,5,9-20] [--local]

A capture folder (written by NL2ModelExport.dll) contains:
    draws.json         every unique draw call of the captured frame(s) with its input-assembler state
    buffers/*.bin      raw GPU buffer contents, named by ID3D11Buffer pointer
    buffers.json       size/bind flags per buffer
    nl3d_buffers.json  the engine's own vertex layout for each vertex buffer (element types + byte offsets)
    cbuffers.bin       vertex-shader constant buffer contents per draw (for world matrices)

Vertex element type codes used by the engine (nl3d::VertexBuffer):
    1 = 4 bytes (packed), 2 = float2, 3 = float3, 4 = float4, 5 = 4-byte colour
The first float3 is the position, the next float3 the normal, the first float2 the UV.
"""
import argparse, json, os, struct, sys
from collections import defaultdict

TYPE_SIZE = {1: 4, 2: 8, 3: 12, 4: 16, 5: 4}
R16_UINT, R32_UINT = 57, 42
TRIANGLELIST, TRIANGLESTRIP = 4, 5


def load(capdir):
    draws = json.load(open(os.path.join(capdir, "draws.json")))
    bufinfo = json.load(open(os.path.join(capdir, "buffers.json")))
    try:
        nl3d = json.load(open(os.path.join(capdir, "nl3d_buffers.json")))
    except Exception:
        nl3d = {}
    cbs = []
    p = os.path.join(capdir, "cbuffers.bin")
    if os.path.exists(p):
        raw = open(p, "rb").read()
        n = struct.unpack_from("<I", raw, 0)[0]; o = 4
        for _ in range(n):
            z = struct.unpack_from("<I", raw, o)[0]; o += 4
            cbs.append(raw[o:o + z]); o += z
    cache = {}

    def buf(name):
        if name not in cache:
            fp = os.path.join(capdir, "buffers", name + ".bin")
            cache[name] = open(fp, "rb").read() if os.path.exists(fp) else None
        return cache[name]
    return draws, bufinfo, nl3d, cbs, buf


def layout_for(nl3d, vbname, stride):
    """Return (pos_off, nrm_off, uv_off) inside a vertex."""
    info = nl3d.get(vbname)
    pos = nrm = uv = None
    if info and info.get("vb") and info["types"]:
        off = 0
        offs = info.get("offsets") or []
        for i, t in enumerate(info["types"]):
            o = offs[i] if i < len(offs) and (offs[i] or i == 0) else off
            if t == 3 and pos is None: pos = o
            elif t == 3 and nrm is None: nrm = o
            elif t == 2 and uv is None: uv = o
            off = o + TYPE_SIZE.get(t, 4)
    if pos is None: pos = 0
    return pos, nrm, uv


def draw_indices(d, buf):
    if d["kind"] in (1, 3):   # non-indexed
        return list(range(d["startVertex"], d["startVertex"] + d["count"]))
    ib = buf(d["ib"])
    if ib is None: return None
    if d["ibFormat"] == R32_UINT:
        fmt, sz = "I", 4
    else:
        fmt, sz = "H", 2
    start = d["ibOffset"] + d["startIndex"] * sz
    n = d["count"]
    if start + n * sz > len(ib): return None
    idx = struct.unpack_from(f"<{n}{fmt}", ib, start)
    return [i + d["baseVertex"] for i in idx]


def triangles(idx, topo):
    if topo == TRIANGLELIST:
        for i in range(0, len(idx) - 2, 3): yield idx[i], idx[i + 1], idx[i + 2]
    elif topo == TRIANGLESTRIP:
        for i in range(len(idx) - 2):
            a, b, c = idx[i], idx[i + 1], idx[i + 2]
            if a == b or b == c or a == c: continue
            yield (a, b, c) if i % 2 == 0 else (b, a, c)


def analyze(capdir):
    draws, bufinfo, nl3d, cbs, buf = load(capdir)
    print(f"{len(draws)} draws, {len(bufinfo)} buffers, {len(nl3d)} engine buffers, {len(cbs)} cb snapshots")
    print(" #  kind topo  count  inst stride vb0(known) ib   bbox-min                bbox-max                 ps/tex")
    for i, d in enumerate(draws):
        vb = d["vb"][0]; stride = d["stride"][0]
        known = "Y" if vb in nl3d else "-"
        bb = ""
        idx = draw_indices(d, buf) if vb != "0" * 16 else None
        data = buf(vb) if vb != "0" * 16 else None
        if idx and data and stride:
            pos, _, _ = layout_for(nl3d, vb, stride)
            base = d["offset"][0]
            xs = []; ys = []; zs = []
            for v in idx[:: max(1, len(idx) // 400)]:
                o = base + v * stride + pos
                if o + 12 <= len(data):
                    x, y, z = struct.unpack_from("<3f", data, o); xs.append(x); ys.append(y); zs.append(z)
            if xs:
                bb = f"({min(xs):8.1f},{min(ys):7.1f},{min(zs):8.1f}) ({max(xs):8.1f},{max(ys):7.1f},{max(zs):8.1f})"
        print(f"{i:3d} {d['kind']:4d} {d['topology']:4d} {d['count']:6d} {d['instances']:5d} {stride:4d}  {vb[-6:]}({known}) "
              f"{d['ib'][-6:]} {bb:48s} {d['ps'][-6:]}/{d['tex0'][-6:]}  inst1={d['vb'][1][-6:]}:{d['stride'][1]}")


def parse_range(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        elif part: out.add(int(part))
    return out


def export(capdir, outpath, min_verts=0, only=None):
    draws, bufinfo, nl3d, cbs, buf = load(capdir)
    vbase = 1
    with open(outpath, "w") as f:
        f.write("# NoLimits 2 capture export (NL2ModelExport)\n")
        for i, d in enumerate(draws):
            if only is not None and i not in only: continue
            if d["topology"] not in (TRIANGLELIST, TRIANGLESTRIP): continue
            vb = d["vb"][0]; stride = d["stride"][0]
            if vb == "0" * 16 or not stride: continue
            data = buf(vb); idx = draw_indices(d, buf)
            if not data or not idx: continue
            used = sorted(set(idx))
            if len(used) < min_verts: continue
            pos, nrm, uv = layout_for(nl3d, vb, stride)
            base = d["offset"][0]
            remap = {}
            f.write(f"o draw_{i:04d}_ps{d['ps'][-6:]}_tex{d['tex0'][-6:]}\n")
            for v in used:
                o = base + v * stride
                if o + stride > len(data): continue
                x, y, z = struct.unpack_from("<3f", data, o + pos)
                f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
                if uv is not None:
                    u, w = struct.unpack_from("<2f", data, o + uv); f.write(f"vt {u:.5f} {1 - w:.5f}\n")
                if nrm is not None:
                    a, b, c = struct.unpack_from("<3f", data, o + nrm); f.write(f"vn {a:.4f} {b:.4f} {c:.4f}\n")
                remap[v] = vbase; vbase += 1
            for a, b, c in triangles(idx, d["topology"]):
                if a in remap and b in remap and c in remap:
                    A, B, C = remap[a], remap[b], remap[c]
                    if uv is not None and nrm is not None: f.write(f"f {A}/{A}/{A} {B}/{B}/{B} {C}/{C}/{C}\n")
                    elif nrm is not None: f.write(f"f {A}//{A} {B}//{B} {C}//{C}\n")
                    elif uv is not None: f.write(f"f {A}/{A} {B}/{B} {C}/{C}\n")
                    else: f.write(f"f {A} {B} {C}\n")
    print("wrote", outpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capdir")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("-o", "--out")
    ap.add_argument("--min-verts", type=int, default=0)
    ap.add_argument("--draws")
    a = ap.parse_args()
    if a.analyze: analyze(a.capdir)
    if a.out: export(a.capdir, a.out, a.min_verts, parse_range(a.draws) if a.draws else None)
