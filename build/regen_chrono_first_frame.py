#!/usr/bin/env python3
"""
Replace each DROID clip's `<clip-id>_chrono.jpg` with a clean first-frame
image (no ghost-stamps composite).

`prepare_clip_simple.py` builds the chrono.jpg as `bg + N convex-hull
stamps` so a single still communicates motion history. For DROID we don't
want that look any more — the prediction trail does the storytelling, and
the stamps clutter the background.

Solution: open each served `<clip-id>.mp4`, grab frame 0, write it as
`<clip-id>_chrono.jpg`. Frame 0 of the served mp4 is by construction the
hist[0] frame in track-fps space (prepare_clip_simple already trimmed +
re-indexed everything to 0-based), so this matches what the trails are
relative to — frame 0 = where the trail starts.
"""
import argparse, os, sys
from pathlib import Path
import cv2


SERVED = Path('/weka/prior-default/jianingz/home/visual/motion-teaser-viz2')
DROID_LABS = ('AUTOLab', 'GuptaLab', 'PennPAL', 'REAL', 'CLVR')


def is_droid(stem: str) -> bool:
    return stem.split('_')[0] in DROID_LABS


def regen(clip_id: str, jpeg_quality: int):
    mp4  = SERVED / 'static/videos' / f'{clip_id}.mp4'
    out  = SERVED / 'static/videos' / f'{clip_id}_chrono.jpg'
    if not mp4.exists():
        print(f'  SKIP (missing mp4): {clip_id}')
        return False
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f'  SKIP (cannot read frame 0): {clip_id}')
        return False
    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    H, W = frame.shape[:2]
    print(f'  wrote {out.name}  ({W}x{H}, {out.stat().st_size//1024} KB)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quality', type=int, default=92,
                    help='JPEG quality (0-100). Default 92 — visually lossless on 720p.')
    ap.add_argument('--all', action='store_true',
                    help='Process every clip in static/data, not just DROID. '
                         'Off by default; we usually only want DROID swapped.')
    args = ap.parse_args()

    data_dir = SERVED / 'static/data'
    clip_ids = []
    for f in data_dir.iterdir():
        if not f.name.endswith('.json'): continue
        # Skip *_pc.bin / *_lastframe / etc.
        stem = f.stem
        if stem.endswith('_pc') or stem.endswith('_lastframe'): continue
        if not args.all and not is_droid(stem): continue
        clip_ids.append(stem)
    clip_ids.sort()
    print(f'regen first-frame chrono for {len(clip_ids)} clips '
          f'({"ALL" if args.all else "DROID only"})')
    n_ok = 0
    for cid in clip_ids:
        print(f'[{cid}]')
        if regen(cid, args.quality): n_ok += 1
    print(f'\nDONE: {n_ok}/{len(clip_ids)} chronos rebuilt')


if __name__ == '__main__':
    main()
