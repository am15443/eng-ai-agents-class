Overview:
This dataset contains object detection results extracted from the video input_video.mp4. Each detection corresponds to a single frame and a single object, including its bounding box, class label, timestamp, and confidence score. This dataset is intended for semantic search and video retrieval tasks based on detected components.

Dataset Schema:
The dataset is stored as a Parquet file: video_detections.parquet. Each row represents one detected object in one frame. The columns are:

Column              Type                Description
video_id	        string	        Unique identifier for the video.
frame_index	        int	            Index of the frame within the video.
timestamp_sec	    float	        Timestamp of the frame in seconds.
class_label	        string	        Object class detected (e.g., car door, wheel).
x_min	            float	        X-coordinate of the top-left corner of the bounding box.
y_min	            float	        Y-coordinate of the top-left corner of the bounding box.
x_max	            float	        X-coordinate of the bottom-right corner of the bounding box.
y_max	            float	        Y-coordinate of the bottom-right corner of the bounding box.
confidence_score	float	        Detection confidence (0–1).
detector_name	    string	        Name of the detection model used (YOLOv8).
model_version	    string	        Version of the detection model.

Usage

The dataset can be used to:
-Search for video segments containing specific components.
-Build retrieval systems that match query images to video content.
-Analyze object presence over time for video understanding.