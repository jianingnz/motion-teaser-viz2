#!/usr/bin/env python3
"""
Update the 16 DROID predictions on the website so they come from the
correct eval results (the previous bake pulled from stale model bundles).

Two groups:

  1. Eight clips with `_object_tNN`  → predictions from
       /weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results/
         rollout5_droid_test_byclass_s*/predictions.jsonl
     The byclass eval is raw 3D only, so we swap pred_3d in place (assembled
     across the per-batch rows whose `point_indices` partition the bundle's
     P points) and *re-project* pred_3d → pred_2d using a (fx,fy,cx,cy)
     calibrated per-clip from the bundle's own (gt_3d, gt_2d) — least-squares
     fits the existing motion5-viz preprocess pipeline to ~5×10⁻⁵ rms in
     normalised image space.

  2. Eight raw / substituted stems → predictions from
       /weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz/static/data/
         modeling_json/droid/droid_v1_ft_f16/<stem>.json
     This is the bundle-shaped output of the same model the user pointed to
     (eval_results/traj3d_droid_v1_ft_f16_per_ds_droid). We re-run
     `prepare_clip_simple.py` against that source so gt/pred/hist/future are
     all from f16. Two of these (44bb9c36, 5d05c5aa_2023-07-13_24400334)
     replace the previous `_object_t20` / `_object_t24` substitutions; the
     stale outputs are deleted from static/.

Final pass: stamp every served bundle with `viewer_defaults.show2DGt =
false` so the 2D video overlay hides the GT trail (only the prediction is
drawn there) — paired with a viewer change in index.html that honours the
flag.
"""
import argparse, glob, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np


M5_JSON = Path('/weka/prior-default/jianingz/home/visual/motion5-viz/static/data/modeling_json/droid/test')
M3_JSON = Path('/weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz/static/data/modeling_json/droid/droid_v1_ft_f16')
EV_BYCLASS = Path('/weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results')
SERVED = Path('/weka/prior-default/jianingz/home/visual/motion-teaser-viz2')

# Raw DROID 1.0.1 release — 1280×720 @ 60 fps. We always source the served mp4
# from here so panel ① stays at 720p (motion5-viz pre-downsampled to 640×360).
DROID_RAW_ROOT = Path('/weka/oe-training-default/jianingz/dataset/droid/1.0.1')

# (stem, t0)  → swap pred from rollout5_droid_test_byclass_*
ROLLOUT5_CLIPS = [
    ('GuptaLab_553d1bd5_2023-04-20-12h-41m-59s_22246076', 33),
    ('PennPAL_c5f808b7_2023-10-09-21h-10m-27s_27085680',  14),
    ('AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334',  27),
    ('REAL_de601749_2023-06-17-11h-34m-24s_20540549',     25),
    ('AUTOLab_0d4edc83_2023-10-21-20h-16m-31s_22008760',  26),
    ('PennPAL_c5f808b7_2023-10-30-00h-50m-14s_25455306',  67),
    ('CLVR_236539bc_2023-06-25-18h-35m-31s_20655732',     19),
    ('AUTOLab_84bd5053_2023-07-14-14h-56m-46s_24400334',  23),
]

# Re-bake from motion3-viz/.../droid_v1_ft_f16/<stem>.json. clip-id == stem.
TRAJ3D_CLIPS = [
    'AUTOLab_0d4edc83_2023-10-21-19h-16m-18s_22008760',
    'AUTOLab_0d4edc83_2023-10-21-19h-46m-27s_22008760',
    'AUTOLab_0d4edc83_2023-10-21-20h-20m-00s_22008760',
    'AUTOLab_0d4edc83_2023-11-03-15h-58m-46s_22008760',
    'AUTOLab_0d4edc83_2023-11-03-16h-52m-04s_22008760',
    'AUTOLab_44bb9c36_2023-11-23-19h-41m-33s_22008760',
    'AUTOLab_5d05c5aa_2023-07-07-18h-52m-04s_22008760',
    'AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334',
]

# Stale clip-ids (replaced by bare-stems above).
TRAJ3D_OLD_CLIPS = [
    'AUTOLab_44bb9c36_2023-11-23-19h-41m-33s_22008760_object_t20',
    'AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334_object_t24',
]

ALL_CLIP_IDS = (
    [f'{s}_object_t{t}' for s, t in ROLLOUT5_CLIPS]
    + TRAJ3D_CLIPS
)


def find_raw_mp4(stem: str) -> Path:
    """Resolve a clip stem (`{lab}_{hex}_{timestamp}_{cam}`) to its raw
    DROID 1.0.1 mp4 by:
      1. Splitting the stem to recover (lab, hex, timestamp, cam).
      2. Globbing /1.0.1/<lab>/{success,failure}/<date>/<run>/metadata_<UUID>.json
         where date = first 10 chars of timestamp, UUID = lab+hex+timestamp.
      3. Returning <run>/recordings/MP4/<cam>.mp4 from the matched run dir.
    The metadata-based lookup keeps us robust against the DROID dataset's
    timestamp-formatted run-dir names (e.g. `Sat_Dec__9_15:45:51_2023`)
    which do not collide with our `2023-12-09-15h-45m-51s` form."""
    parts = stem.split('_')
    if len(parts) < 4:
        raise RuntimeError(f'cannot parse droid stem: {stem}')
    lab, hexid, ts, cam = parts[0], parts[1], parts[2], parts[3]
    date = ts[:10]   # YYYY-MM-DD
    uuid = f'{lab}+{hexid}+{ts}'
    for split in ('success', 'failure'):
        for meta in (DROID_RAW_ROOT / lab / split / date).glob(f'*/metadata_{uuid}.json'):
            run_dir = meta.parent
            mp4 = run_dir / 'recordings' / 'MP4' / f'{cam}.mp4'
            if mp4.exists():
                return mp4
    raise RuntimeError(f'no raw mp4 found for {stem} '
                       f'(searched {DROID_RAW_ROOT}/{lab}/<succ|fail>/{date}/*/metadata_{uuid}.json)')


def stem_to_video(stem: str) -> str:
    """`AUTOLab_5d05c5aa_2023-07-13-...s_24400334` →
       `AUTOLab+5d05c5aa+2023-07-13-...s_24400334`. The eval files use `+` as
       the lab/hex/timestamp separator and `_` only before the cam serial."""
    parts = stem.split('_')
    return '+'.join(parts[:3]) + '_' + parts[3]


def fit_projection(gt3: np.ndarray, gt2: np.ndarray):
    """Solve (fx_norm, fy_norm, cx_norm, cy_norm) so that
       u = fx_norm * X / Z + cx_norm,  v = fy_norm * Y / Z + cy_norm.
    Pulls every (X,Y,Z,u,v) sample with finite values + Z>0.05 out of the
    bundle. Residuals are typically <1e-4 on motion5-viz bundles since the
    bundle's own gt_2d came from the exact same projection."""
    X = gt3[..., 0]; Y = gt3[..., 1]; Z = gt3[..., 2]
    u = gt2[..., 0]; v = gt2[..., 1]
    m = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
         & np.isfinite(u) & np.isfinite(v) & (Z > 0.05))
    if int(m.sum()) < 4:
        raise RuntimeError('not enough valid (gt_3d, gt_2d) samples to fit projection')
    xz = (X / Z)[m]; yz = (Y / Z)[m]; un = u[m]; vn = v[m]
    Au = np.stack([xz, np.ones_like(xz)], 1); su, *_ = np.linalg.lstsq(Au, un, rcond=None)
    Av = np.stack([yz, np.ones_like(yz)], 1); sv, *_ = np.linalg.lstsq(Av, vn, rcond=None)
    return float(su[0]), float(sv[0]), float(su[1]), float(sv[1])


def project(pred3: np.ndarray, fxn, fyn, cxn, cyn) -> np.ndarray:
    F, P, _ = pred3.shape
    out = np.full((F, P, 2), np.nan, dtype=np.float64)
    valid = (np.isfinite(pred3[..., 0]) & np.isfinite(pred3[..., 1])
             & np.isfinite(pred3[..., 2]) & (pred3[..., 2] > 0.05))
    out[..., 0] = np.where(valid, fxn * pred3[..., 0] / pred3[..., 2] + cxn, np.nan)
    out[..., 1] = np.where(valid, fyn * pred3[..., 1] / pred3[..., 2] + cyn, np.nan)
    return out


def to_jsonable(arr):
    """np.ndarray → nested Python list with `None` for NaN (matches the
    motion5-viz bundle convention for invisible-track entries)."""
    if arr.ndim == 0:
        v = float(arr)
        return None if not np.isfinite(v) else v
    return [to_jsonable(sub) for sub in arr]


def gather_byclass(stem: str, t0: int, obj: str = 'object'):
    target = stem_to_video(stem)
    rows = []
    for fp in sorted(EV_BYCLASS.glob('rollout5_droid_test_byclass_s*/predictions.jsonl')):
        with open(fp) as f:
            for ln in f:
                d = json.loads(ln)
                if d.get('video') == target and d.get('obj') == obj and d.get('t0') == t0:
                    rows.append(d)
    return rows


def swap_rollout5(stem: str, t0: int):
    clip_id = f'{stem}_object_t{t0}'
    served = SERVED / 'static/data' / f'{clip_id}.json'
    if not served.exists():
        raise RuntimeError(f'served bundle missing: {served}')
    bundle = json.loads(served.read_text())
    cfg = bundle['configs'][0]
    gt3 = np.array(cfg['gt_3d'], dtype=np.float64)
    gt2 = np.array(cfg['gt_2d'], dtype=np.float64)
    F, P, _ = gt3.shape
    nh = cfg['n_hist']

    rows = gather_byclass(stem, t0)
    if not rows:
        raise RuntimeError(f'no byclass rows for {stem} t0={t0}')

    # Sanity: union of point_indices across batches must cover the bundle's P.
    union = set()
    for r in rows:
        union.update(r['point_indices'])
    if len(union) < P:
        print(f'  WARN: {clip_id}: byclass union covers {len(union)}/{P} points')

    # Future-frame count check: bundle's future = F - nh, byclass's = pred_raw_combined.shape[1].
    F_fut_expected = F - nh
    F_fut_eval = len(rows[0]['pred_raw_combined'][0])
    if F_fut_eval != F_fut_expected:
        print(f'  WARN: {clip_id}: bundle fut={F_fut_expected} vs byclass fut={F_fut_eval}; '
              f'will use min')
    F_use = min(F_fut_expected, F_fut_eval)

    # Assemble pred_3d = (F, P, 3): hist replicates gt_3d (so pred[hist]=gt[hist]),
    # future fills from each batch's pred_raw_combined indexed by `point_indices`.
    new_pred3 = np.full((F, P, 3), np.nan, dtype=np.float64)
    new_pred3[:nh] = gt3[:nh]
    for r in rows:
        pi = r['point_indices']
        pr = np.array(r['pred_raw_combined'], dtype=np.float64)  # (P_batch, F_fut, 3)
        for j, p in enumerate(pi):
            if p < 0 or p >= P:
                continue
            new_pred3[nh:nh + F_use, p] = pr[j, :F_use]

    fxn, fyn, cxn, cyn = fit_projection(gt3, gt2)
    new_pred2 = project(new_pred3, fxn, fyn, cxn, cyn)

    cfg['pred_3d'] = to_jsonable(new_pred3)
    cfg['pred_2d'] = to_jsonable(new_pred2)
    served.write_text(json.dumps(bundle))
    n_finite = int(np.sum(np.isfinite(new_pred3[..., 0])))
    print(f'  wrote {clip_id}.json  pred_3d finite={n_finite}/{F*P}  '
          f'fxn={fxn:.3f} fyn={fyn:.3f} cxn={cxn:.3f} cyn={cyn:.3f}')


def remove_old_clip_files(clip_id: str):
    deleted = 0
    for d in [SERVED / 'static/data', SERVED / 'static/videos']:
        if not d.exists():
            continue
        for f in list(d.iterdir()):
            n = f.name
            if (n == f'{clip_id}.json'
                or n == f'{clip_id}.mp4'
                or n.startswith(f'{clip_id}_pc')
                or n.startswith(f'{clip_id}_chrono')
                or n == f'{clip_id}_clipframes' or n == f'{clip_id}_fullframes'):
                if f.is_file():
                    f.unlink(); deleted += 1
                elif f.is_dir():
                    shutil.rmtree(f); deleted += 1
    print(f'  removed {deleted} files for stale clip-id {clip_id}')


def _run_prepare_clip(src_json: Path, clip_id: str, stem: str):
    """Invoke prepare_clip_simple.py with src_mp4 pinned to the raw DROID
    1.0.1 720p mp4 (so the served panel-① mp4 isn't downsampled)."""
    if not src_json.exists():
        raise RuntimeError(f'missing src json: {src_json}')
    src_mp4 = find_raw_mp4(stem)
    cmd = [sys.executable, str(SERVED / 'build/prepare_clip_simple.py'),
           '--src-json', str(src_json),
           '--src-mp4',  str(src_mp4),
           '--out-dir',  str(SERVED),
           '--clip-id',  clip_id]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f'prepare_clip_simple failed for {clip_id} rc={rc}')


def rebake_traj3d_v1(stem: str):
    """Pull a clean bundle from motion3-viz/.../droid_v1_ft_f16/<stem>.json
    (which already has gt/pred for the f16 model) and write it to the
    served dir at 720p."""
    _run_prepare_clip(M3_JSON / f'{stem}.json', stem, stem)


def rebake_rollout5_at_720p(stem: str, t0: int):
    """For the rollout5 group we keep the motion5-viz bundle's gt + history
    indexing but re-bake the served mp4/chrono/pc.bin from the 720p raw
    DROID mp4. After this, swap_rollout5() overwrites pred_3d/pred_2d in
    the served JSON with the byclass eval predictions."""
    clip_id = f'{stem}_object_t{t0}'
    src_json = M5_JSON / f'{clip_id}.json'
    _run_prepare_clip(src_json, clip_id, stem)


def stamp_viewer_defaults(clip_ids):
    """Set `viewer_defaults.show2DGt = false` in every served bundle so the
    2D video / chrono overlays hide GT for DROID. The 3D panels still show
    GT — this only affects the canvas-on-top-of-mp4 trail."""
    for cid in clip_ids:
        served = SERVED / 'static/data' / f'{cid}.json'
        if not served.exists():
            print(f'  SKIP (missing): {cid}')
            continue
        bundle = json.loads(served.read_text())
        vd = bundle.get('viewer_defaults') or {}
        vd['show2DGt'] = False
        bundle['viewer_defaults'] = vd
        served.write_text(json.dumps(bundle))
    print(f'  stamped show2DGt=false on {len(clip_ids)} bundles')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-rollout5', action='store_true')
    ap.add_argument('--skip-traj3d',   action='store_true')
    ap.add_argument('--skip-densify',  action='store_true',
                    help='Skip the dense PC re-bake at the end. Useful when iterating.')
    ap.add_argument('--skip-viewer-defaults', action='store_true')
    args = ap.parse_args()

    if not args.skip_rollout5:
        # Two-pass: first re-bake the rollout5 bundles from the raw 720p
        # mp4 (motion5-viz's pre-baked mp4 was 640×360), then overwrite
        # pred_3d/pred_2d with the byclass eval. Step 1 resets pred to
        # whatever motion5-viz had stored; step 2 stamps the correct
        # byclass pred on top.
        print('\n=== rollout5 step 1/2: re-bake 8 clips at 720p ===')
        for stem, t0 in ROLLOUT5_CLIPS:
            print(f'[{stem}_object_t{t0}]')
            rebake_rollout5_at_720p(stem, t0)
        print('\n=== rollout5 step 2/2: swap pred from byclass eval ===')
        for stem, t0 in ROLLOUT5_CLIPS:
            print(f'[{stem}_object_t{t0}]')
            swap_rollout5(stem, t0)

    if not args.skip_traj3d:
        print('\n=== remove obsolete substituted clip-ids ===')
        for old in TRAJ3D_OLD_CLIPS:
            remove_old_clip_files(old)
        print('\n=== re-bake traj3d_v1 group (8 clips, droid_v1_ft_f16) ===')
        for stem in TRAJ3D_CLIPS:
            print(f'[{stem}]')
            rebake_traj3d_v1(stem)

    if not args.skip_densify:
        print('\n=== re-densify scene PCs ===')
        rc = subprocess.run([sys.executable, str(SERVED / 'build/dense_pc_droid16.py')]).returncode
        if rc != 0:
            print('!!! dense_pc_droid16 failed; PCs may be sparse')

    if not args.skip_viewer_defaults:
        print('\n=== stamp viewer_defaults.show2DGt=false on all 16 ===')
        stamp_viewer_defaults(ALL_CLIP_IDS)

    print('\nALL DONE')


if __name__ == '__main__':
    main()
