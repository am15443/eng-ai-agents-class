from huggingface_hub import create_repo

# Replace with your HF username
repo_id = "anubhutimathur/HW2_AI_am15443"

create_repo(repo_id=repo_id, repo_type="dataset", private=True)
print(f"Dataset created: {repo_id}")