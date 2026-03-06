# read_parquet.py
import pandas as pd
import os

# Path to the Parquet file
parquet_file = "index/video_detections.parquet"
excel_file = "index/video_detections.xlsx"

# Check if file exists
if not os.path.exists(parquet_file):
    raise FileNotFoundError(f"Parquet file not found at {parquet_file}")

# Read the Parquet file
df = pd.read_parquet(parquet_file)

# Print basic info
print("Parquet file loaded successfully.")
print(f"Number of rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Show first 10 rows
print("\nFirst 10 rows:")
print(df.head(10))

# Export to Excel
df.to_excel(excel_file, index=False)
print(f"Data exported to Excel file: {excel_file}")