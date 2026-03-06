from ultralytics import YOLO
import cv2
import os
import pandas as pd

def detect_video_frames(frame_dir, video_id="input_video", conf_threshold=0.5, fps_sampling=1):
    model = YOLO("yolov8n.pt")

    rows = []
    frame_files = sorted(os.listdir(frame_dir))

    for frame_index, fname in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        results = model(frame)[0]
        timestamp_sec = frame_index * fps_sampling

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            class_id = int(box.cls[0])
            class_label = model.names[class_id]
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            rows.append({
                "video_id": video_id,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "class_label": class_label,
                "bounding_box": (x_min, y_min, x_max, y_max),
                "confidence_score": conf,
                "detector_name": "YOLOv8",
                "model_version": "yolov8n"
            })

    df = pd.DataFrame(rows)

    os.makedirs("index", exist_ok=True)
    output_path = "index/video_detections.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} detections to {output_path}")