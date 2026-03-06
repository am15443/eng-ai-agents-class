from src.extract_frames import extract_frames

if __name__ == "__main__":
    video_path = "data/input_video.mp4"
    output_dir = "data/frames"
    fps = 1
    start_time_sec = 120
    duration_sec = 45

    extract_frames(video_path, output_dir, fps, start_time_sec, duration_sec)
    