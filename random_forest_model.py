import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

def train_random_forest_model(
        X_train,
        y_train,
        n_estimators,
        max_depth,
        min_samples_leaf
):

    model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth= max_depth,
    min_samples_leaf= min_samples_leaf,
    n_jobs= -2,
    random_state = 42
    )

    model.fit(X_train, y_train)

    return model


def check_model_performance(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):
    performance_list = []

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)

    test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    performance_list.append({
        "model" : model_name,
        "train_precision" : train_precision,
        "train_recall" : train_recall,
        "train_f1" : train_f1,
        "test_precision" : test_precision,
        "test_recall" : test_recall,
        "test_f1" : test_f1,
    })

    return pd.DataFrame(performance_list)


def evaluate_thresholds(
        model,
        model_name,
        thresholds,
        X_test,
        y_test,
):

    results = []
    fraud_probs = model.predict_proba(X_test)[:, 1]
    for threshold in thresholds:
        y_pred = (fraud_probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        results.append({
            "model" : model_name,
            "threshold" : threshold,
            "TN" : tn,
            "FP" : fp,
            "FN" : fn,
            "TP" : tp,
            "precision" : precision,
            "recall" : recall,
            "f1" : f1,
        })
    return pd.DataFrame(results)

def plot_metrics_by_threshold(results_df, model_name, save_path):
    model_results = results_df[results_df["model"] == model_name]

    plt.figure(figsize=(10, 6))

    plt.plot(
        model_results["threshold"],
        model_results["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        model_results["threshold"],
        model_results["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        model_results["threshold"],
        model_results["f1"],
        marker="o",
        label="F1 Score"
    )

    plt.title(f"{model_name} - Metrics by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fp_fn_by_threshold(results_df, model_name, save_path):
    model_results = results_df[results_df["model"] == model_name]

    plt.figure(figsize=(10, 6))

    plt.plot(
        model_results["threshold"],
        model_results["FP"],
        marker="o",
        label="False Positives"
    )

    plt.plot(
        model_results["threshold"],
        model_results["FN"],
        marker="o",
        label="False Negatives"
    )

    plt.title(f"{model_name} - FP and FN by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


#=================
#load data
#=================
pd.set_option("display.max_columns", None)
df = pd.read_parquet("data/paysim.parquet")


#=================
#prepare X/y
#=================
X = df[[
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]]

y = df['isFraud']

X = pd.get_dummies(
    X,
    columns=["type"],
    drop_first=True,
    dtype = int
)


#=================
# train/test split
#=================
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)


#=================
#train model
#=================
controlled_forest_model = train_random_forest_model(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
)

stronger_forest_model = train_random_forest_model(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=20,
)


#=================
#model check
#=================
controlled_forest_model_performance = check_model_performance(
    controlled_forest_model,
    "controlled_forest_model",
    X_train,
    y_train,
    X_test,
    y_test
)

stronger_forest_model_performance = check_model_performance(
    stronger_forest_model,
    "stronger_forest_model",
    X_train,
    y_train,
    X_test,
    y_test
)


#=================
#evaluate
#=================
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results_controlled_forest_model = evaluate_thresholds(
    controlled_forest_model,
    "controlled_forest_model",
    thresholds,
    X_test,
    y_test
)

results_stronger_forest_model = evaluate_thresholds(
    stronger_forest_model,
    "stronger_forest_model",
    thresholds,
    X_test,
    y_test
)


#=================
#merge
#=================
performance_df = pd.concat(
    [
        controlled_forest_model_performance,
        stronger_forest_model_performance
    ],
    ignore_index = True
)

threshold_results_df = pd.concat(
    [
        results_controlled_forest_model,
        results_stronger_forest_model
    ],
    ignore_index = True
)


#=================
#candidate results
#=================
random_forest_candidate_model = threshold_results_df[
    (threshold_results_df["FP"] <= 100)&
    (threshold_results_df["recall"] >= 0.70)
].sort_values("f1", ascending = False)


#=================
#save
#=================
os.makedirs("outputs/random_forest", exist_ok=True)

performance_df.to_csv("outputs/random_forest/random_forest_model_performance.csv", index=False)
threshold_results_df.to_csv("outputs/random_forest/random_forest_threshold_results.csv", index=False)
random_forest_candidate_model.to_csv("outputs/random_forest/random_forest_candidate_models.csv", index=False)


#=================
#plot
#=================
plot_metrics_by_threshold(
    threshold_results_df,
    "controlled_forest_model",
    "outputs/random_forest/controlled_forest_metrics_by_threshold.png"
)

plot_fp_fn_by_threshold(
    threshold_results_df,
    "controlled_forest_model",
    "outputs/random_forest/controlled_forest_fp_fn_by_threshold.png"
)

plot_metrics_by_threshold(
    threshold_results_df,
    "stronger_forest_model",
    "outputs/random_forest/stronger_forest_metrics_by_threshold.png"
)

plot_fp_fn_by_threshold(
    threshold_results_df,
    "stronger_forest_model",
    "outputs/random_forest/stronger_forest_fp_fn_by_threshold.png"
)