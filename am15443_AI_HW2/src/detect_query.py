from datasets import load_dataset
from ultralytics import YOLO
import numpy as np

def detect_query_components():
    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")
    model = YOLO("yolov8n.pt")

    query_detections = []

    for i, item in enumerate(ds):
        image = np.array(item["image"])  # Convert PIL → numpy
        results = model(image)[0]

        classes = [model.names[int(box.cls[0])] for box in results.boxes]

        query_detections.append({
            "query_index": i,
            "classes": classes
        })

    return query_detections