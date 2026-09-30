#!/usr/bin/env python3
"""Ask the injected NL2ModelExport DLL to capture the next N rendered frames.
    py capture.py [frames] [nocb] [noscan] [full] [inst] [out_dir] [--port N]     default: 1 frame -> ..\\captures\\<timestamp>
"full" = whole park in one frame: culling, occlusion and LOD distance switched off for the capture (DLL v10+).
The port is read from the newest *.port file next to the DLL (build\\) unless --port is given.
"""
import glob, os, socket, sys, time
args = [a for a in sys.argv[1:]]
port = None
if "--port" in args:
    i = args.index("--port"); port = int(args[i + 1]); del args[i:i + 2]
here = os.path.dirname(os.path.abspath(__file__))
if port is None:
    ports = sorted(glob.glob(os.path.join(here, "..", "build", "*.port")), key=os.path.getmtime)
    port = int(open(ports[-1]).read().strip()) if ports else 15153
flags = [a for a in args if a in ("nocb", "noscan", "full", "inst")]
args = [a for a in args if a not in flags]
frames = int(args[0]) if args else 1
out = os.path.abspath(args[1]) if len(args) > 1 else os.path.abspath(os.path.join(here, "..", "captures", time.strftime("%Y%m%d_%H%M%S")))
os.makedirs(os.path.dirname(out), exist_ok=True)
s = socket.create_connection(("127.0.0.1", port), timeout=330)
s.sendall(("CAPTURE %d %s%s\n" % (frames, "".join(f + " " for f in flags), out)).encode())
print(f"[port {port}]", s.recv(4096).decode().strip())
