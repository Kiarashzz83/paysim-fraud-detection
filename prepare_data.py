import pandas as pd
pd.set_option("display.max_columns", None)
df = pd.read_csv("data/paysim.csv")
df.to_parquet("data/paysim.parquet")