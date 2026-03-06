import cv2
import os

def extract_frames(video_path, output_dir, fps=1, start_time_sec=0, duration_sec=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time_sec * video_fps)
    end_frame = total_frames
    if duration_sec is not None:
        end_frame = min(total_frames, start_frame + int(duration_sec * video_fps))

    frame_interval = max(1, round(video_fps / fps))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < start_frame:
            frame_count += 1
            continue
        if frame_count >= end_frame:
            break

        if (frame_count - start_frame) % frame_interval == 0:
            cv2.imwrite(
                f"{output_dir}/frame_{saved_count:05d}.jpg",
                frame
            )
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")