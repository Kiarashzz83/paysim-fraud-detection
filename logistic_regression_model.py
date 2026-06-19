import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import os

#=================
#functions
#=================
def train_logistic_model(model_type, X_train, y_train):
    if model_type == "baseline":
        model = LogisticRegression(max_iter=1000)

    elif model_type == "balanced":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    else:
        raise ValueError("model_type must be 'baseline' or 'balanced'")

    model.fit(X_train, y_train)
    return model


def evaluate_thresholds(model_name, model, X_test, y_test, thresholds):
    results = []

    fraud_probs = model.predict_proba(X_test)[:, 1]

    for threshold in thresholds:
        y_pred = (fraud_probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        results.append({
            "model": model_name,
            "threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)


def plot_metrics_by_threshold(results_df, plot_title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        results_df["threshold"],
        results_df["recall"],
        marker="o",
        label="Recall"
    )

    ax.plot(
        results_df["threshold"],
        results_df["f1"],
        label="F1-score"
    )

    ax.set_title(plot_title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_fp_fn_by_threshold(results_df, plot_title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        results_df["threshold"],
        results_df["FP"],
        marker="o",
        label="False Positives"
    )

    ax.plot(
        results_df["threshold"],
        results_df["FN"],
        marker="o",
        label="False Negatives"
    )

    ax.set_title(plot_title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300 ,bbox_inches="tight")
    plt.show()
    plt.close()
#=================
#load data
#=================
pd.set_option("display.max_columns", None)
df = pd.read_parquet("data/paysim.parquet")

#=================
#prepare X/y
#=================
X = df[['step',
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
       'newbalanceDest']]
y = df['isFraud']
X = pd.get_dummies(
    X,
    columns=['type'],
    drop_first=True,
    dtype=int
)


#=================
# train/test split
#=================
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify= y
)

#=================
#scale
#=================
numeric_cols = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])



#=================
#train model
#=================
baseline_model = train_logistic_model(
    "baseline",
    X_train_scaled,
    y_train
)

balanced_model = train_logistic_model(
    "balanced",
    X_train_scaled,
    y_train
)

#=================
#evaluate thresholds
#=================
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results_baseline = evaluate_thresholds(
    "baseline",
    baseline_model,
    X_test_scaled,
    y_test,
    thresholds
)

results_balanced = evaluate_thresholds(
    "balanced",
    balanced_model,
    X_test_scaled,
    y_test,
    thresholds
)


#=================
#make df of results
#=================

results_df = pd.concat([results_baseline, results_balanced], ignore_index=True)




#=================
#candidate results
#=================
candidate_models = results_df[
    (results_df["FP"] <= 500) &
    (results_df["recall"] >= 0.45)
].sort_values("f1", ascending=False)


#=================
#save results
#=================
os.makedirs("outputs/logistic_regression", exist_ok=True)

results_df.to_csv(
    "outputs/logistic_regression/logistic_regression_threshold_results.csv",
    index=False
)

candidate_models.to_csv(
    "outputs/logistic_regression/logistic_regression_candidate_models.csv",
    index=False
)


#=================
#plot
#=================
plot_metrics_by_threshold(
    results_baseline,
    "Baseline Logistic Regression: Metrics by Threshold",
    "outputs/logistic_regression/logistic_regression_baseline_thresholds.png"
)

plot_metrics_by_threshold(
    results_balanced,
    "Balanced Logistic Regression: Metrics by Threshold",
    "outputs/logistic_regression/logistic_regression_balanced_thresholds.png"
)

plot_fp_fn_by_threshold(
    results_baseline,
    "Baseline Logistic Regression: FP and FN by Threshold",
    "outputs/logistic_regression/logistic_regression_baseline_fp_fn.png"
)

plot_fp_fn_by_threshold(
    results_balanced,
    "Balanced Logistic Regression: FP and FN by Threshold",
    "outputs/logistic_regression/logistic_regression_balanced_fp_fn.png"
)

print("Logistic Regression results and plots saved successfully.")
print(candidate_models)