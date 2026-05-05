#!/usr/bin/env python3
"""
Bake the 16 selected DROID clips (curated by jianingz) into motion-teaser-viz2.

Mapping:
  - 8 stems with `_object_tNN` come from motion5-viz/static/data/modeling_json/droid/test/
  - 6 raw stems (5×0d4edc83 + 5d05c5aa_2023-07-07) come from
    motion3-viz/static/data/modeling_json/droid/2mix_droid_molmospaces_p8_h3_f8/
  - 2 substituted stems: AUTOLab_44bb9c36_..._t20 and AUTOLab_5d05c5aa_2023-07-13_..._t24
    (user listed without t#; we pick the available motion5-viz config)
  - mp4 sources: motion5-viz/.../droid/<stem>.mp4 when present, else
    motion3-viz/.../droid/<stem>.mp4

`prepare_clip_simple.py` is invoked once per clip; it produces:
  static/data/<clip-id>.json
  static/data/<clip-id>_pc.bin
  static/videos/<clip-id>.mp4
  static/videos/<clip-id>_chrono.jpg

`pred_2d/3d` already equals `gt_2d/3d` on the 3 history frames in source JSONs,
so the "first 3 GT frames prepended to prediction" property comes for free.
"""
import argparse, subprocess, sys
from pathlib import Path

M5  = Path("/weka/prior-default/jianingz/home/visual/motion5-viz")
M3  = Path("/weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz")
M5_JSON_DIR = M5 / "static/data/modeling_json/droid/test"
M3_JSON_DIR = M3 / "static/data/modeling_json/droid/droid_v1_ft_f16"
M5_VID_DIR  = M5 / "static/videos/modeling/droid"
M3_VID_DIR  = M3 / "static/videos/modeling/droid"

# Raw DROID 1.0.1 (720p) — the canonical source. We resolve a clip stem
# (`{lab}_{hex}_{timestamp}_{cam}`) to its run dir via the metadata file, so
# we don't depend on the downsampled motion5-viz / motion3-viz mp4 mirrors.
DROID_RAW_ROOT = Path("/weka/oe-training-default/jianingz/dataset/droid/1.0.1")


def find_raw_mp4(stem: str) -> Path:
    parts = stem.split("_")
    if len(parts) < 4:
        raise RuntimeError(f"cannot parse droid stem: {stem}")
    lab, hexid, ts, cam = parts[0], parts[1], parts[2], parts[3]
    date = ts[:10]
    uuid = f"{lab}+{hexid}+{ts}"
    for split in ("success", "failure"):
        for meta in (DROID_RAW_ROOT / lab / split / date).glob(f"*/metadata_{uuid}.json"):
            mp4 = meta.parent / "recordings" / "MP4" / f"{cam}.mp4"
            if mp4.exists():
                return mp4
    raise RuntimeError(
        f"no raw mp4 for {stem} (searched {DROID_RAW_ROOT}/{lab}/<split>/{date}/*)"
    )

# (clip_id, src_json_dir, src_vid_dir, base_stem, suffix)
# where src_json = src_json_dir/<base_stem><suffix>.json
# and   src_mp4  = src_vid_dir /<base_stem>.mp4
# clip_id = base_stem + suffix  (kept identical to the source JSON stem so
# rebuild_pc_droid_dense can parse uuid+cam back from it later if needed)
CLIPS = [
    ("GuptaLab_553d1bd5_2023-04-20-12h-41m-59s_22246076",        M5_JSON_DIR, M5_VID_DIR, "_object_t33"),
    ("PennPAL_c5f808b7_2023-10-09-21h-10m-27s_27085680",         M5_JSON_DIR, M5_VID_DIR, "_object_t14"),
    ("AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334",         M5_JSON_DIR, M5_VID_DIR, "_object_t27"),
    ("REAL_de601749_2023-06-17-11h-34m-24s_20540549",            M5_JSON_DIR, M5_VID_DIR, "_object_t25"),
    ("AUTOLab_0d4edc83_2023-10-21-20h-16m-31s_22008760",         M5_JSON_DIR, M5_VID_DIR, "_object_t26"),
    ("PennPAL_c5f808b7_2023-10-30-00h-50m-14s_25455306",         M5_JSON_DIR, M5_VID_DIR, "_object_t67"),
    ("CLVR_236539bc_2023-06-25-18h-35m-31s_20655732",            M5_JSON_DIR, M5_VID_DIR, "_object_t19"),
    ("AUTOLab_84bd5053_2023-07-14-14h-56m-46s_24400334",         M5_JSON_DIR, M5_VID_DIR, "_object_t23"),
    # raw stems (no _object_tNN) — sourced from motion3-viz
    ("AUTOLab_0d4edc83_2023-10-21-19h-16m-18s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    ("AUTOLab_0d4edc83_2023-10-21-19h-46m-27s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    ("AUTOLab_0d4edc83_2023-10-21-20h-20m-00s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    ("AUTOLab_0d4edc83_2023-11-03-15h-58m-46s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    ("AUTOLab_0d4edc83_2023-11-03-16h-52m-04s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    # bare-stem clips that were previously substituted with rollout5 t# variants;
    # now sourced from the same droid_v1_ft_f16 config as the other six raw stems.
    ("AUTOLab_44bb9c36_2023-11-23-19h-41m-33s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    ("AUTOLab_5d05c5aa_2023-07-07-18h-52m-04s_22008760",         M3_JSON_DIR, M3_VID_DIR, ""),
    ("AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334",         M3_JSON_DIR, M3_VID_DIR, ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/weka/prior-default/jianingz/home/visual/motion-teaser-viz2"))
    ap.add_argument("--only", default=None,
                    help="If given, only bake the clip whose id starts with this prefix.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    prepare = here / "prepare_clip_simple.py"
    if not prepare.exists():
        raise RuntimeError(f"missing {prepare}")

    failures = []
    for base, jdir, vdir, suff in CLIPS:
        clip_id = base + suff
        if args.only and not clip_id.startswith(args.only):
            continue
        src_json = jdir / f"{clip_id}.json"
        if not src_json.exists():
            print(f"!!! MISSING json: {src_json}", flush=True)
            failures.append((clip_id, "json"))
            continue
        # Always source the mp4 from the raw 720p DROID release. The vdir
        # parameter is no longer used — kept in CLIPS for documentation.
        try:
            src_mp4 = find_raw_mp4(base)
        except RuntimeError as e:
            print(f"!!! {e}", flush=True)
            failures.append((clip_id, "mp4"))
            continue
        cmd = [sys.executable, str(prepare),
               "--src-json", str(src_json),
               "--src-mp4",  str(src_mp4),
               "--out-dir",  str(args.out_dir),
               "--clip-id",  clip_id]
        print(f"\n=== bake {clip_id} ===", flush=True)
        print("  $", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((clip_id, f"rc={rc}"))

    print("\n----- DONE -----")
    if failures:
        print("FAILURES:")
        for c, w in failures:
            print(f"  {c}  ({w})")
        sys.exit(1)
    else:
        print(f"all {len(CLIPS)} clips baked OK")


if __name__ == "__main__":
    main()
