"""Per-clip cosmetic overrides applied AFTER the standard build pipeline
(swap_droid_predictions.py / regen_chrono_first_frame.py).

These tweaks are not part of the prediction-source pipeline — they exist so
specific clips render the way we want for the teaser figure. Re-run this
script whenever the upstream bundle is regenerated.

Currently configured:

  - REAL_de601749_2023-06-17-11h-34m-24s_20540549_object_t25
        Hide the last 6 frames of pred_2d (NaN-out) so the 2D-overlay
        prediction trail stops short of the end. The 3D pred and the GT
        remain untouched.

The viewer's drawChronoOverlay supports an optional `chrono.trail_start_frame`
field (default 0); the helper `set_chrono_start_frame` is kept here for future
use even though no clips currently set a non-zero start.
"""

import json
from pathlib import Path

import cv2

SERVED = Path('/weka/prior-default/jianingz/home/visual/motion-teaser-viz2')
DATA = SERVED / 'static/data'
VIDEOS = SERVED / 'static/videos'


def hide_pred_2d_tail(clip_id: str, hide_last_n: int):
    p = DATA / f'{clip_id}.json'
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.loads(p.read_text())
    cfgs = d['configs']
    F = d['num_frames']
    keep_to = F - hide_last_n
    nan_pt = [None, None]
    for c in cfgs:
        P = len(c['pred_2d'][0])
        for t in range(keep_to, F):
            c['pred_2d'][t] = [list(nan_pt) for _ in range(P)]
    p.write_text(json.dumps(d))
    print(f'  {clip_id}: pred_2d[{keep_to}:{F}] hidden ({hide_last_n} frames)')


def set_chrono_start_frame(clip_id: str, frame_idx: int, jpeg_quality: int = 92):
    """Re-render the chrono.jpg using `frame_idx` of the served mp4 and stamp
    `chrono.trail_start_frame = frame_idx` into the bundle JSON.

    Together these make panel ② open at the chosen frame and have the
    chrono trail (+ start markers) begin from there. drawVideoOverlay is
    not touched, so panel ① still plays from frame 0.
    """
    json_path = DATA / f'{clip_id}.json'
    mp4_path = VIDEOS / f'{clip_id}.mp4'
    chrono_jpg = VIDEOS / f'{clip_id}_chrono.jpg'
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not mp4_path.exists():
        raise FileNotFoundError(mp4_path)

    # 1) Frame extraction. Seek to frame_idx; abort if mp4 is too short.
    cap = cv2.VideoCapture(str(mp4_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= n_frames:
        cap.release()
        raise ValueError(f'{clip_id}: frame_idx={frame_idx} >= n_frames={n_frames}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'{clip_id}: cannot read frame {frame_idx}')
    cv2.imwrite(str(chrono_jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # 2) JSON stamp.
    d = json.loads(json_path.read_text())
    d.setdefault('chrono', {})['trail_start_frame'] = frame_idx
    json_path.write_text(json.dumps(d))

    H, W = frame.shape[:2]
    print(f'  {clip_id}: chrono frame={frame_idx}  ({W}x{H})  '
          f'chrono.jpg={chrono_jpg.stat().st_size//1024}KB  '
          f'JSON.chrono.trail_start_frame={frame_idx}')


def main():
    print('Applying per-clip cosmetic overrides...')

    # REAL clip — pred_2d trail stops 6 frames before the end.
    hide_pred_2d_tail(
        'REAL_de601749_2023-06-17-11h-34m-24s_20540549_object_t25',
        hide_last_n=6,
    )

    print('done.')


if __name__ == '__main__':
    main()
