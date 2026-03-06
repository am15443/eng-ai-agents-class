from src.retrieval import retrieve_segments
from src.detect_query import detect_query_components

if __name__ == "__main__":
    query_detections = detect_query_components()

    for q in query_detections:
        df_segments = retrieve_segments(q["classes"])
        print(f"\nQuery {q['query_index']} segments:")
        print(df_segments)