
import pandas as pd
from pathlib import Path

# Find CSV files in the project folder
csv_files = list(Path(".").glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found in the project folder.")
else:
    print("CSV files found:", [f.name for f in csv_files])

    # Read the first CSV file
    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("\nDataset filename:", file_path.name)
    print("Dataset shape (rows, columns):", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values in each column:")
    print(df.isnull().sum().to_string())

    print("\nUnique values in each column:")
    for column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts().head(10).to_string())
