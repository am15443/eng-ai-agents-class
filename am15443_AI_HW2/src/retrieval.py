import pandas as pd

def retrieve_segments(query_classes, video_parquet="index/video_detections.parquet"):
    df = pd.read_parquet(video_parquet)
    segments = []

    for class_label in query_classes:
        subset = df[df["class_label"] == class_label].sort_values("timestamp_sec")
        if subset.empty:
            continue

        # Find contiguous segments
        current_start = None
        prev_ts = None
        supporting_count = 0

        for ts in subset["timestamp_sec"]:
            if current_start is None:
                current_start = ts
                supporting_count = 1
            elif ts - prev_ts > 1:  # gap larger than sampling interval
                segments.append({
                    "start_timestamp": current_start,
                    "end_timestamp": prev_ts,
                    "class_label": class_label,
                    "number_of_supporting_detections": supporting_count
                })
                current_start = ts
                supporting_count = 1
            else:
                supporting_count += 1
            prev_ts = ts

        # final segment
        if current_start is not None:
            segments.append({
                "start_timestamp": current_start,
                "end_timestamp": prev_ts,
                "class_label": class_label,
                "number_of_supporting_detections": supporting_count
            })

    return pd.DataFrame(segments)