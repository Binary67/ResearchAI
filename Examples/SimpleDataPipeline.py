from Problem.Harness.DataPipeline import KaggleDataLoader

loader = KaggleDataLoader()

# Download a dataset by its Kaggle slug (e.g. "zillow/zecon")
frames = loader.download("yasserh/housing-prices-dataset")

# `frames` is a dict mapping filenames to DataFrames
for name, df in frames.items():
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
