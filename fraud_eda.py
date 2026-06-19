import pandas as pd
pd.set_option("display.max_columns", None)
df = pd.read_parquet("data/paysim.parquet")

print(df.head())
print()
print()
print(df['isFraud'].value_counts())
print(df["isFlaggedFraud"].value_counts())
fraud_count_by_type = df.groupby("type").agg(
    total_transactions=("type", "count"),
    fraud_count=("isFraud", "sum"),
    fraud_percent=("isFraud", "mean"),
    flagged_fraud_count = ("isFlaggedFraud", "sum")
).reset_index()
fraud_count_by_type["fraud_percent"] = fraud_count_by_type['fraud_percent'] * 100
print(fraud_count_by_type)


df_compare = df[[
    'isFraud',
    'isFlaggedFraud'
]].copy()
true_negative = ((df_compare['isFraud'] == 0) & (df_compare['isFlaggedFraud'] == 0)).sum()
false_negative = ((df_compare['isFraud'] == 1) & (df_compare['isFlaggedFraud'] == 0)).sum()
true_positive = ((df_compare['isFraud'] == 1) & (df_compare['isFlaggedFraud'] == 1)).sum()
false_positive = ((df_compare['isFraud'] == 0) & (df_compare['isFlaggedFraud'] == 1)).sum()

print(f'True negative: {true_negative}')
print(f'False negative: {false_negative}')
print(f'True positive: {true_positive}')
print(f'False positive: {false_positive}')

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
print(f'Precision: {precision}')
print(f'Recall: {recall}')

print((df == 0).sum())

zero_org_check_fraud = df.groupby("isFraud").agg(
    oldbalanceOrg_zero_percent=("oldbalanceOrg", lambda x: (x == 0).mean() * 100),
    newbalanceOrig_zero_percent=("newbalanceOrig", lambda x: (x == 0).mean() * 100)
)

print(zero_org_check_fraud)

zero_dest_check_fraud = df.groupby("isFraud").agg(
    oldbalanceDest_zero_percent=("oldbalanceDest", lambda x: (x == 0).mean() * 100),
    newbalanceDest_zero_percent=("newbalanceDest", lambda x: (x == 0).mean() * 100)
)

print(zero_dest_check_fraud)

amount_check_fraud = df.groupby("isFraud").agg(
    count = ("amount", "count"),
    mean = ("amount", "mean"),
    median = ("amount", "median"),
    min = ("amount", "min"),
    max = ("amount", "max")
)

print(amount_check_fraud)


df["origin_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
df["destination_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

origin_balance_error_fraud_check = df.groupby('isFraud').agg(
    count=("origin_balance_error", "count"),
    mean=("origin_balance_error", "mean"),
    median=("origin_balance_error", "median"),
    min=("origin_balance_error", "min"),
    max=("origin_balance_error", "max")
)

destination_balance_error_fraud_check = df.groupby('isFraud').agg(
    count=("destination_balance_error", "count"),
    mean=("destination_balance_error", "mean"),
    median=("destination_balance_error", "median"),
    min=("destination_balance_error", "min"),
    max=("destination_balance_error", "max")
)
print(origin_balance_error_fraud_check)
print(destination_balance_error_fraud_check)

