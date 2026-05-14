"""
영상에서 손 움직임 분석 데이터 추출.
MediaPipe 대신 OpenCV 기반으로 프레임 밝기/움직임 변화를 추출하여
finger tapping의 주기성과 amplitude를 분석.
"""

import cv2
import json
import os
import numpy as np

VIDEO_DIR = '/Users/moonie/Desktop/시각화수업'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')

VIDEOS = [
    {'side': 'left', 'folder': '7. Finger_tapping_left'},
    {'side': 'right', 'folder': '8. FInger_tapping_right'},
]

CAMERAS = ['Camera1.mp4', 'Camera2.mp4', 'Camera3.mp4',
           'Camera4.mp4', 'Camera5.mp4', 'Camera6.mp4']


def extract_motion_from_video(video_path, sample_every=1):
    """Frame differencing 기반 움직임 추출."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_gray = None
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ROI: 중앙 영역 (손이 주로 있는 곳)
        roi_y1, roi_y2 = height // 4, height * 3 // 4
        roi_x1, roi_x2 = width // 4, width * 3 // 4
        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

        motion_intensity = 0.0
        if prev_gray is not None:
            prev_roi = prev_gray[roi_y1:roi_y2, roi_x1:roi_x2]
            diff = cv2.absdiff(roi, prev_roi)
            motion_intensity = float(np.mean(diff))

        frames.append({
            'frame': frame_idx,
            'time': round(frame_idx / fps, 4),
            'motion_intensity': round(motion_intensity, 4),
            'roi_brightness': round(float(np.mean(roi)), 2),
        })

        prev_gray = gray
        frame_idx += 1

    cap.release()

    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_s': round(total_frames / fps, 2),
        'frames': frames,
    }


def main():
    all_data = {}

    for video_cfg in VIDEOS:
        side = video_cfg['side']
        folder = video_cfg['folder']
        all_data[side] = {}

        for camera in CAMERAS:
            video_path = os.path.join(VIDEO_DIR, folder, camera)
            if not os.path.exists(video_path):
                print(f"  [SKIP] {video_path}")
                continue

            print(f"Processing {side} {camera}...")
            data = extract_motion_from_video(video_path, sample_every=2)
            data['camera'] = camera
            data['side'] = side

            # Peak detection for tap counting
            motions = [f['motion_intensity'] for f in data['frames']]
            if motions:
                threshold = np.mean(motions) + np.std(motions)
                peaks = []
                for i in range(1, len(motions) - 1):
                    if motions[i] > threshold and motions[i] > motions[i-1] and motions[i] > motions[i+1]:
                        peaks.append(i)
                data['tap_peak_indices'] = peaks
                data['estimated_taps'] = len(peaks)

                if len(peaks) > 1:
                    times = [data['frames'][p]['time'] for p in peaks]
                    intervals = np.diff(times)
                    data['mean_tap_interval'] = round(float(np.mean(intervals)), 4)
                    data['tap_interval_cv'] = round(float(np.std(intervals) / np.mean(intervals)), 4) if np.mean(intervals) > 0 else 0
                else:
                    data['mean_tap_interval'] = 0
                    data['tap_interval_cv'] = 0

            all_data[side][camera] = data
            print(f"  → {data['total_frames']} frames, {data['estimated_taps']} taps detected")

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'video_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(all_data, f)

    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
