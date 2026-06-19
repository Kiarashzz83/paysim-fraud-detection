import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

def train_decision_tree_model(model_type, X_train, y_train):
    if model_type == "raw_tree_model":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "controlled_tree_model":
        model = DecisionTreeClassifier(
            random_state= 42,
            max_depth= 10,
            min_samples_leaf = 50,
        )
    else:
        raise ValueError("model_type must be 'raw_tree_model' or 'controlled_tree_model'")
    model.fit(X_train, y_train)
    return model



def check_model_performance(model,model_name, X_train, y_train, X_test, y_test):
    check_list = []

    tree_depth = model.get_depth()
    leaves_count = model.get_n_leaves()

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)

    test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    check_list.append({
        'model': model_name,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'tree_depth': tree_depth,
        'leaves_count': leaves_count
    })
    return pd.DataFrame(check_list)



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



def metrics_by_threshold_plot(result_df, plot_title, save_path):

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(
        result_df["threshold"],
        result_df["precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        result_df["threshold"],
        result_df["recall"],
        marker="o",
        label="Recall"
    )

    ax.plot(
        result_df["threshold"],
        result_df["f1"],
        label="F1"

    )

    ax.set_title(plot_title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle = ":", alpha = 0.6)
    ax.legend(loc = "best")

    plt.tight_layout()
    plt.savefig(save_path, dpi = 300)
    plt.show()
    plt.close()



def fp_fn_by_threshold_plot(result_df, plot_title, save_path):
    fig, ax = plt.subplots(figsize = (8, 5))

    ax.plot(
        result_df["threshold"],
        result_df["FN"],
        marker="o",
        label="FN"
    )

    ax.plot(
        result_df["threshold"],
        result_df["FP"],
        marker="o",
        label="FP"
    )

    ax.set_title(plot_title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
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
X = df[[
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]]

y = df["isFraud"]

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
raw_tree_model = train_decision_tree_model(
    "raw_tree_model",
    X_train,
    y_train
)

controlled_tree_model = train_decision_tree_model(
    "controlled_tree_model",
    X_train,
    y_train
)


#=================
#model check
#=================
raw_performance = check_model_performance(
    raw_tree_model,
    "raw_tree_model",
    X_train,
    y_train,
    X_test,
    y_test
)

controlled_performance = check_model_performance(
    controlled_tree_model,
    "controlled_tree_model",
    X_train,
    y_train,
    X_test,
    y_test
)


#=================
#evaluate
#=================
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results_raw_tree_model = evaluate_thresholds(
    'raw_tree_model',
    raw_tree_model,
    X_test,
    y_test,
    thresholds
)

results_controlled_tree_model = evaluate_thresholds(
    'controlled_tree_model',
    controlled_tree_model,
    X_test,
    y_test,
    thresholds
)


#=================
#results
#=================
performance_check_df = pd.concat([
    raw_performance,
    controlled_performance],
    ignore_index=True)

thresholds_results_df = pd.concat(
    [results_raw_tree_model,
     results_controlled_tree_model],
    ignore_index=True
)


#=================
##candidate results
#=================
candidate_models = thresholds_results_df[
    (thresholds_results_df['FP'] <= 500) &
    (thresholds_results_df['recall'] >= 0.70)
].sort_values("f1", ascending=False)


#=================
#save
#=================
os.makedirs('outputs/decision_tree', exist_ok = True)

performance_check_df.to_csv(
    'outputs/decision_tree/decision_tree_model_performance.csv',
                            index = False
)

thresholds_results_df.to_csv(
    'outputs/decision_tree/decision_tree_threshold_results.csv',
    index = False
)

candidate_models.to_csv(
    'outputs/decision_tree/decision_tree_candidate_models.csv',
    index = False
)


#=================
#plot
#=================
metrics_by_threshold_plot(
    results_raw_tree_model,
    "Raw tree metrics by threshold",
    "outputs/decision_tree/raw_tree_metrics_by_threshold.png"
)

metrics_by_threshold_plot(
    results_controlled_tree_model,
    "Controlled tree metrics by threshold",
    "outputs/decision_tree/controlled_tree_metrics_by_threshold.png"
)

fp_fn_by_threshold_plot(
    results_raw_tree_model,
    "Raw tree FP/FN by threshold",
    "outputs/decision_tree/raw_tree_fp_fn_by_threshold.png"
)

fp_fn_by_threshold_plot(
    results_controlled_tree_model,
    "Controlled tree FP/FN by threshold",
    "outputs/decision_tree/controlled_tree_fp_fn_by_threshold.png"
)