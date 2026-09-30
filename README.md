# NL2 Model Export

Get the 3D models of a NoLimits 2 coaster park out of the game, so you can turn your ride into a physical,
3D-printed model.

A small DLL is injected into the running game. It records what the game draws and writes it back out as
glTF (`.glb`) or OBJ files: track, supports, footers, catwalks, station, scenery and trains. You can then
open them in Blender (or any 3D tool), scale them and prepare them for printing.

> **Want a model you are allowed to sell?** Use the spline route instead:
> [nl2-blender-track-builder](https://github.com/zm2283145/nl2-blender-track-builder) rebuilds the ride in
> Blender from your own exported track spline and supports. With its *Procedural Parts Only* mode (the
> default without a game asset pack), every part is original geometry.

---

## ⚠️ Read this first: personal use only

**Models ripped with this DLL must not be sold or redistributed.**

- Everything this tool exports comes from the **in-game models and textures**. Those were built by, and belong
  to, the developers of NoLimits 2. Capturing them does not make them yours.
- You may use the exports **for yourself**, for example to 3D-print a model of your own ride for your desk.
- **Do not sell** the exported files or prints made from them. Do not upload them to model stores or sharing
  sites, and do not offer them as a print service.
- To make something you can sell, use the spline route with
  [nl2-blender-track-builder](https://github.com/zm2283145/nl2-blender-track-builder), and read its licensing
  notes.

This project is not affiliated with or endorsed by the makers of NoLimits 2. NoLimits is a trademark of its
owners. Ride and manufacturer names belong to their owners too. Injecting code into a game may be against its
EULA or terms of service: check them, and use this at your own risk. None of this is legal advice.

---

## Requirements

- Windows and NoLimits 2 (the Steam build, `nolimits2stm.exe`). The DLL uses fixed addresses inside the game
  executable, so a different game build needs updated addresses. It checks the code before patching
  anything, but other builds are not supported.
- Python 3 with `numpy` (`pip install numpy`).
- To build from source: MinGW-w64, for example from MSYS2 (`pacman -S mingw-w64-x86_64-gcc`).
- Blender 4.2+ to view and prepare the exports (optional).

## Build

```bat
build.cmd
```

This produces the following in `build\`:

| File | What it is |
|---|---|
| `NL2ModelExport.dll` | The capture DLL |
| `nl2vfsdump.dll` | Optional helper that copies files out of the game's own data packages (see below) |
| `nl2inject.exe` | Loads a DLL into the running game |

A DLL that is loaded in the game can't be overwritten. If you rebuild while the game is running, use
`build.cmd 2`, `build.cmd 3`, … to get `NL2ModelExport_v2.dll` and so on.

Prebuilt binaries are attached to the GitHub releases.

## Export a park

1. Start NoLimits 2 and open your park.
2. Inject the capture DLL:
   ```bat
   build\nl2inject.exe build\NL2ModelExport.dll
   ```
   The DLL listens on `127.0.0.1:15153` (or the next free port, written to `build\<dll>.port`) and logs to
   `build\<dll>.log`.
3. Bring the game window to the front, then capture the whole park in one frame. `full` turns off culling
   and LOD for that frame, so everything is drawn at its finest detail.
   ```bat
   py tools\capture.py 1 full
   ```
   This writes `captures\<timestamp>\`, typically 50–150 MB.
4. *Optional: bolts and nuts across the whole park.* The game only draws bolts within about 50–60 m of the
   camera. Run the harvester and ride or fly along the whole coaster, then press Ctrl-C:
   ```bat
   py tools\harvest_instances.py captures\harvest.pkl
   ```
   Don't reload the park while harvesting.
5. Convert the capture to models:
   ```bat
   py tools\nl2_export.py captures\<timestamp> -o export [--instances captures\harvest.pkl] [--format glb|obj|both]
   ```

### Output

| File | Contents |
|---|---|
| `layout.glb` | Track, supports, footers, catwalks, stairs, station and queue, plus the trains where they were parked |
| `layout_no_trains.glb` | The same without trains. **This is usually the one to print.** |
| `scenery.glb` | Scenery and terrain (`--no-scenery` / `--no-terrain` leave it out) |
| `train_front_car.glb`, `train_car.glb` | One car of each type in local space (`--all-cars` writes every car) |
| `report.txt` | What was found and classified |

## Preparing a print

1. In Blender, use **File → Import → glTF 2.0** and import `layout_no_trains.glb`.
2. Scale to your target, for example 1:100 (scale 0.01; the exports are in metres).
3. Thin parts such as rails, cross-ties and handrails are usually too thin at small scales. Thicken them with
   a Solidify modifier, or remove them.
4. Use the **3D Print Toolbox** add-on to check for non-manifold geometry and wall thickness.
5. Split the model into pieces that fit your printer bed, and add pins or keys to join them.

## Dumping game files for the Blender add-on (optional)

[nl2-blender-track-builder](https://github.com/zm2283145/nl2-blender-track-builder) can optionally use a few
meshes (ties, connectors, footers) from **your own** game install, for a closer personal-use replica. It does
not need them: without them it builds fully procedural parts. `nl2vfsdump.dll` copies them out of the game's
data packages:

1. Write the file list: `python build_assets.py --list build\nl2vfsdump.txt` (the script is in the add-on
   repo).
2. Inject: `build\nl2inject.exe build\nl2vfsdump.dll`. It reads `nl2vfsdump.txt` next to the DLL, writes the
   files to `build\vfs\data\…`, logs to `nl2vfsdump.log` and unloads itself.

The dumped files belong to the NoLimits 2 developers: keep them on your own machine.

## How it works (short version)

NoLimits 2 builds its track geometry on the fly and keeps it only in GPU buffers. The DLL hooks Direct3D 11
(`Present` and the `Draw*` calls). While a capture is armed, it records every draw with its buffers,
constant buffers, shaders and textures, and reads the engine's own vertex-layout objects from memory.
`nl2_export.py` (numpy only) then:

- decodes the meshes;
- recovers each draw's world transform and material/paint;
- drops shadow, reflection and LOD duplicates;
- sorts the parts into ride, scenery and trains;
- writes glTF or OBJ.

## License

The source code is MIT-licensed (see [LICENSE](LICENSE)). The license covers this code only. It gives no
rights to NoLimits 2 or to anything captured from it.
