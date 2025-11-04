import pandas as pd
import polars as pl
df = pd.read_parquet('./data/dapo-math-17k.parquet')

# Remove duplicates
pl_df = pl.from_pandas(df).unique(subset=["data_source", "prompt", "ability", "reward_model"])

# Count number of reward_models per prompt
pl_df = pl_df.with_columns(
    pl.col("reward_model").n_unique().over("prompt").alias("n_rm")
)

# Keep only prompts with one reward_model
cleaned = pl_df.filter(pl.col("n_rm") == 1).drop("n_rm")

# Convert back to pandas and save
cleaned_df = cleaned.to_pandas()
cleaned_df.to_parquet('./data/dapo-math-17k.parquet')
print(f"Before: {len(df)} rows, After: {len(cleaned_df)} rows")