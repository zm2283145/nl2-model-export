#!/usr/bin/env python3
"""Collect instanced detail parts (bolts, nuts, support connectors) from the whole park.

The game only draws these within ~50-60 m of the camera, and the one-shot "full" capture can't lift that limit.
Their instance records hold exact world matrices, so they can be gathered from many captures while the camera
moves (ride the coaster, or fly along it) and merged onto a one-shot capture:

    py harvest_instances.py <out.pkl> [--count N] [--every SEC]
        ... ride / fly around the park while it runs (Ctrl-C or --count ends it) ...
    py nl2_export.py <one-shot capture> --instances <out.pkl> -o <out_dir>

Needs DLL v11+ (CAPTURE ... inst). Keeps the first capture (a whole frame) next to out.pkl (<out>_src0) for the
meshes and materials, plus any later one that brings a mesh not seen yet, reads only the instance records of the
others and deletes them. Don't reload the park while harvesting: meshes are matched by GPU buffer address.
"""
import argparse, glob, os, pickle, shutil, socket, sys, time
import numpy as np
import nl2_export as E

def capture(out, port, inst=True):
    s = socket.create_connection(("127.0.0.1", port), timeout=330)
    s.sendall(("CAPTURE 1 full noscan %s%s\n" % ("inst " if inst else "", out)).encode())
    r = s.recv(4096).decode().strip(); s.close()
    if not r.startswith("OK"): raise RuntimeError(r)

def instances(path):
    """{mesh key: set of world matrices (3x4, rounded to 0.1 mm)} of every instanced draw"""
    cap = E.Capture(path); got = {}
    for i, x in enumerate(cap.d):
        if x["kind"] not in (2, 3) or x.get("inst", -1) < 0: continue
        mk = (x["vb"][0], x["ib"], x["startIndex"], x["count"], x["baseVertex"])
        s = got.setdefault(mk, set())
        for W in E.instance_worlds(cap, i): s.add(tuple(np.round(W[:3, :].ravel(), 4)))
    return got

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("--count", type=int, default=0, help="stop after N captures (0 = until Ctrl-C)")
    ap.add_argument("--every", type=float, default=0.0, help="seconds between captures (default: back to back)")
    ap.add_argument("--port", type=int)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    port = a.port
    if port is None:
        ports = sorted(glob.glob(os.path.join(here, "..", "build", "*.port")), key=os.path.getmtime)
        port = int(open(ports[-1]).read().strip()) if ports else 15153
    base = os.path.splitext(os.path.abspath(a.out))[0]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for d in glob.glob(base + "_src*") + glob.glob(base + "_tmp"): shutil.rmtree(d, ignore_errors=True)
    acc = {}; sources = []; have = set(); n = 0
    def save():
        pickle.dump({"sources": sources, "inst": {k: np.array(sorted(v)).reshape(-1, 3, 4) for k, v in acc.items()}},
                    open(a.out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    try:
        while not a.count or n < a.count:
            t0 = time.time()
            path = base + "_tmp"
            if os.path.exists(path): shutil.rmtree(path)
            capture(path, port, inst=n > 0)   # the first one is a whole frame: the exporter needs it to tell materials from overlay passes
            got = instances(path)
            for k, v in got.items(): acc.setdefault(k, set()).update(v)
            new = set(got) - have
            if new:   # meshes the kept captures don't have: keep this one too for their vertex data
                keep = base + "_src%d" % len(sources); os.rename(path, keep); sources.append(keep); have |= set(got)
            else:
                shutil.rmtree(path, ignore_errors=True)
            n += 1; save()
            print(f"[{n}] {sum(len(v) for v in acc.values())} unique instances of {len(acc)} meshes"
                  + (f", kept as source ({len(new)} new meshes)" if new else ""), flush=True)
            if a.every: time.sleep(max(0.0, a.every - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    save(); print(f"saved {a.out} ({n} captures, {len(sources)} kept as sources)")

if __name__ == "__main__":
    main()
