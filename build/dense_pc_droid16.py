#!/usr/bin/env python3
"""
Re-bake every DROID `*_pc.bin` in motion-teaser-viz2 at stride-1 dense
sampling (full 360×640 depth grid, no random cap).

Wraps `rebuild_pc_droid_dense.py`. Passes `--uuid` and `--cam` explicitly
because the wrapped script's auto-parser only recognises clip-ids that
start with `AUTOLab` — we have GuptaLab / PennPAL / REAL / CLVR too.

Per-clip mapping mirrors `bake_droid16.py`:
  - 8 stems with `_object_tNN` → motion5-viz/static/data/modeling_json/droid/test/<stem>.json
  - 6 raw stems              → motion3-viz/.../2mix_droid_molmospaces_p8_h3_f8/<stem>.json
  - 2 substituted t# stems   → motion5-viz/.../<base>_object_t{20,24}.json

Outputs (overwrites in-place):
  static/data/<clip-id>_pc.bin     ← stride-1 PC (~10–230 k points)
  static/data/<clip-id>.json       ← pc_bin.n_points patched by the wrapped script
"""
import argparse, subprocess, sys
from pathlib import Path

M5_JSON = Path("/weka/prior-default/jianingz/home/visual/motion5-viz/static/data/modeling_json/droid/test")
M3_JSON = Path("/weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz/static/data/modeling_json/droid/droid_v1_ft_f16")
M5_VID  = Path("/weka/prior-default/jianingz/home/visual/motion5-viz/static/videos/modeling/droid")
M3_VID  = Path("/weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz/static/videos/modeling/droid")

# Each row: (base_stem, src_json_dir, src_vid_dir, json_suffix)
# clip_id = base_stem + json_suffix  (matches what bake_droid16.py wrote)
CLIPS = [
    ("GuptaLab_553d1bd5_2023-04-20-12h-41m-59s_22246076",  M5_JSON, M5_VID, "_object_t33"),
    ("PennPAL_c5f808b7_2023-10-09-21h-10m-27s_27085680",   M5_JSON, M5_VID, "_object_t14"),
    ("AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334",   M5_JSON, M5_VID, "_object_t27"),
    ("REAL_de601749_2023-06-17-11h-34m-24s_20540549",      M5_JSON, M5_VID, "_object_t25"),
    ("AUTOLab_0d4edc83_2023-10-21-20h-16m-31s_22008760",   M5_JSON, M5_VID, "_object_t26"),
    ("PennPAL_c5f808b7_2023-10-30-00h-50m-14s_25455306",   M5_JSON, M5_VID, "_object_t67"),
    ("CLVR_236539bc_2023-06-25-18h-35m-31s_20655732",      M5_JSON, M5_VID, "_object_t19"),
    ("AUTOLab_84bd5053_2023-07-14-14h-56m-46s_24400334",   M5_JSON, M5_VID, "_object_t23"),
    ("AUTOLab_0d4edc83_2023-10-21-19h-16m-18s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_0d4edc83_2023-10-21-19h-46m-27s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_0d4edc83_2023-10-21-20h-20m-00s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_0d4edc83_2023-11-03-15h-58m-46s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_0d4edc83_2023-11-03-16h-52m-04s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_44bb9c36_2023-11-23-19h-41m-33s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_5d05c5aa_2023-07-07-18h-52m-04s_22008760",   M3_JSON, M3_VID, ""),
    ("AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334",   M3_JSON, M3_VID, ""),
]


def parse_uuid_cam(base: str):
    """`AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334` →
       uuid='AUTOLab+5d05c5aa+2023-07-13-10h-59m-53s', cam='24400334'.
       Works for any DROID lab prefix (AUTOLab/GuptaLab/PennPAL/REAL/CLVR/...)."""
    parts = base.split("_")
    if len(parts) < 4:
        raise RuntimeError(f"unexpected base stem: {base}")
    uuid = "+".join(parts[:3])
    cam  = parts[3]
    return uuid, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/weka/prior-default/jianingz/home/visual/motion-teaser-viz2"))
    ap.add_argument("--only", default=None,
                    help="Only rebake clip ids starting with this prefix.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    rebuild = here / "rebuild_pc_droid_dense.py"
    if not rebuild.exists():
        raise RuntimeError(f"missing {rebuild}")

    failures = []
    for base, jdir, vdir, suff in CLIPS:
        clip_id = base + suff
        if args.only and not clip_id.startswith(args.only):
            continue
        src_json = jdir / f"{clip_id}.json"
        src_mp4  = vdir / f"{base}.mp4"
        uuid, cam = parse_uuid_cam(base)
        if not src_json.exists():
            print(f"!!! MISSING json: {src_json}"); failures.append((clip_id, "json")); continue
        if not src_mp4.exists():
            print(f"!!! MISSING mp4:  {src_mp4}");  failures.append((clip_id, "mp4"));  continue
        cmd = [sys.executable, str(rebuild),
               "--src-json", str(src_json),
               "--src-mp4",  str(src_mp4),
               "--out-dir",  str(args.out_dir),
               "--clip-id",  clip_id,
               "--uuid",     uuid,
               "--cam",      cam,
               "--subsample", "1",
               "--max-points", "0"]
        print(f"\n=== dense {clip_id}  uuid={uuid}  cam={cam} ===")
        print("  $", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((clip_id, f"rc={rc}"))

    print("\n----- DONE -----")
    if failures:
        print("FAILURES:")
        for c, w in failures: print(f"  {c}  ({w})")
        sys.exit(1)
    else:
        print(f"all {len(CLIPS)} clips re-baked at stride 1")


if __name__ == "__main__":
    main()
