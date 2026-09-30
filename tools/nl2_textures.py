"""Decode the .dds files NL2ModelExport captures (DX10 header, top mip only) to RGBA numpy arrays / PNG.
Only needs numpy (PNG writing uses zlib from the standard library)."""
import struct, zlib
import numpy as np

def _unpack565(c):
    r = ((c >> 11) & 31) * 255 // 31; g = ((c >> 5) & 63) * 255 // 63; b = (c & 31) * 255 // 31
    return np.stack([r, g, b], -1).astype(np.int32)

def _bc1_color(blk, alpha1=True):
    # blk: (n, 8) uint8 -> (n, 16, 4) uint8
    c0 = blk[:, 0].astype(np.int32) | (blk[:, 1].astype(np.int32) << 8)
    c1 = blk[:, 2].astype(np.int32) | (blk[:, 3].astype(np.int32) << 8)
    idx = blk[:, 4:8].copy().view(np.uint32)[:, 0]
    p0, p1 = _unpack565(c0), _unpack565(c1)
    four = (c0 > c1) | (not alpha1)
    p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    pal = np.stack([p0, p1, p2, p3], 1)                       # (n,4,3)
    a = np.full((len(blk), 4), 255, np.int32)
    if alpha1: a[:, 3] = np.where(four, 255, 0)
    sel = (idx[:, None] >> (2 * np.arange(16))[None]) & 3       # (n,16)
    rgb = np.take_along_axis(pal, sel[..., None].repeat(3, -1), 1)
    al = np.take_along_axis(a, sel, 1)
    return np.concatenate([rgb, al[..., None]], -1).astype(np.uint8)

def _bc4_channel(blk):
    # blk: (n, 8) -> (n, 16) uint8
    a0 = blk[:, 0].astype(np.int32); a1 = blk[:, 1].astype(np.int32)
    bits = np.zeros(len(blk), np.uint64)
    for k in range(6): bits |= blk[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    sel = ((bits[:, None] >> (3 * np.arange(16, dtype=np.uint64))[None]) & np.uint64(7)).astype(np.int32)
    j = np.arange(8)[None]
    six = ((8 - j) * a0[:, None] + (j - 1) * a1[:, None]) // 7     # a0 > a1: 6 interpolated values
    four = ((6 - j) * a0[:, None] + (j - 1) * a1[:, None]) // 5     # else 4 interpolated + 0 + 255
    pal = np.where((a0 > a1)[:, None], six, four)
    pal[:, 0] = a0; pal[:, 1] = a1
    small = ~(a0 > a1)
    pal[small, 6] = 0; pal[small, 7] = 255
    return np.take_along_axis(np.clip(pal, 0, 255), sel, 1).astype(np.uint8)

def _blocks_to_image(px, w, h):
    # px: (bh*bw, 16, C) -> (h, w, C)
    bw, bh = (w + 3) // 4, (h + 3) // 4; C = px.shape[-1]
    img = px.reshape(bh, bw, 4, 4, C).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, C)
    return img[:h, :w]

def decode(path):
    """returns (rgba uint8 array (h,w,4), srgb flag) or None if the format is not supported"""
    raw = open(path, "rb").read()
    h, w = struct.unpack_from("<II", raw, 12)
    fmt = struct.unpack_from("<I", raw, 128)[0]
    data = np.frombuffer(raw, np.uint8, offset=148)
    srgb = fmt in (72, 75, 78, 91, 93, 99)
    bw, bh = (w + 3) // 4, (h + 3) // 4
    if fmt in (87, 88, 91, 90, 92, 93):                         # B8G8R8A8 / B8G8R8X8
        pitch = len(data) // h
        a = data[:pitch * h].reshape(h, pitch)[:, :w * 4].reshape(h, w, 4)
        out = a[..., [2, 1, 0, 3]].copy()
        if fmt in (88, 92, 93): out[..., 3] = 255
        return out, srgb
    if fmt in (28, 29, 30):                                     # R8G8B8A8
        pitch = len(data) // h
        return data[:pitch * h].reshape(h, pitch)[:, :w * 4].reshape(h, w, 4).copy(), fmt == 29
    if fmt in (61, 62, 63, 64, 65):                             # R8
        pitch = len(data) // h; g = data[:pitch * h].reshape(h, pitch)[:, :w]
        return np.dstack([g, g, g, np.full_like(g, 255)]), False
    if fmt in (49, 50, 51, 52):                                 # R8G8
        pitch = len(data) // h; a = data[:pitch * h].reshape(h, pitch)[:, :w * 2].reshape(h, w, 2)
        return np.dstack([a[..., 0], a[..., 1], np.zeros_like(a[..., 0]), np.full_like(a[..., 0], 255)]), False
    if fmt in (70, 71, 72):                                     # BC1
        pitch = len(data) // bh; blk = data[:pitch * bh].reshape(bh, pitch)[:, :bw * 8].reshape(-1, 8)
        return _blocks_to_image(_bc1_color(blk), w, h), srgb
    if fmt in (73, 74, 75):                                     # BC2
        pitch = len(data) // bh; blk = data[:pitch * bh].reshape(bh, pitch)[:, :bw * 16].reshape(-1, 16)
        px = _bc1_color(blk[:, 8:], alpha1=False)
        al = blk[:, :8]; a = np.stack([(al >> 0) & 15, (al >> 4) & 15], -1).reshape(-1, 16) * 17
        px[..., 3] = a
        return _blocks_to_image(px, w, h), srgb
    if fmt in (76, 77, 78):                                     # BC3
        pitch = len(data) // bh; blk = data[:pitch * bh].reshape(bh, pitch)[:, :bw * 16].reshape(-1, 16)
        px = _bc1_color(blk[:, 8:], alpha1=False); px[..., 3] = _bc4_channel(blk[:, :8])
        return _blocks_to_image(px, w, h), srgb
    if fmt in (79, 80, 81):                                     # BC4
        pitch = len(data) // bh; blk = data[:pitch * bh].reshape(bh, pitch)[:, :bw * 8].reshape(-1, 8)
        g = _bc4_channel(blk)[..., None]
        px = np.concatenate([g, g, g, np.full_like(g, 255)], -1)
        return _blocks_to_image(px, w, h), False
    if fmt in (82, 83, 84):                                     # BC5 (usually a normal map: x,y)
        pitch = len(data) // bh; blk = data[:pitch * bh].reshape(bh, pitch)[:, :bw * 16].reshape(-1, 16)
        r = _bc4_channel(blk[:, :8]); g = _bc4_channel(blk[:, 8:])
        x = r.astype(np.float32) / 127.5 - 1; y = g.astype(np.float32) / 127.5 - 1
        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1)); b = ((z + 1) * 127.5).astype(np.uint8)
        px = np.stack([r, g, b, np.full_like(r, 255)], -1)
        return _blocks_to_image(px, w, h), False
    return None

def png_bytes(rgba):
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + \
           chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
