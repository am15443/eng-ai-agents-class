from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="index/video_detections.parquet",
    path_in_repo="video_detections.parquet",
    repo_id="anubhutimathur/HW2_AI_am15443",
    repo_type="dataset"
)