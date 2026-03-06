from src.detect_video import detect_video_frames

if __name__ == "__main__":
    frame_dir = "data/frames"
    detect_video_frames(
        frame_dir=frame_dir,
        video_id="input_video",
        conf_threshold=0.5,
        fps_sampling=1
    )