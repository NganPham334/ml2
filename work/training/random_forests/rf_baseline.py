import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, mean_absolute_error, r2_score, \
    confusion_matrix, accuracy_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit



# ==========================================
# HELPER: CHRONOLOGICAL SPLIT
# ==========================================
def get_splits(df, target_col):
    """Splits data 80/20 chronologically."""
    # 1. Sort by time
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

    # 2. Separate X and y
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. Drop Metadata/Leakage (Datetime is never a feature)
    if 'datetime' in X.columns:
        X = X.drop(columns=['datetime'])

    # 4. Split 80/20
    split_idx = int(len(X) * 0.8)
    return X.iloc[:split_idx], X.iloc[split_idx:], y[:split_idx], y[split_idx:]


def get_tscv(n_splits=5):
    return TimeSeriesSplit(n_splits=n_splits)



# ==========================================
# 1. NON-PCA CLASSIFICATION
# ==========================================
def run_standard_classification():
    print(f"\n{'=' * 40}\nTASK 1: Standard Classification (No PCA, Tuned)\n{'=' * 40}")
    try:
        df = pd.read_csv("../../S4_lagged_class.csv")
    except FileNotFoundError:
        print("❌ File 'S4_lagged_class.csv' not found. Skipping.")
        return

    le = LabelEncoder()
    precip_cols = [c for c in df.columns if c.startswith('preciptype')]
    for col in precip_cols:
        df[col] = le.fit_transform(df[col])

    cond_cols = [c for c in df.columns if c.startswith("conditions")]
    for col in cond_cols:
        df[col] = le.fit_transform(df[col])

    leakage = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
               'dew', 'humidity', 'precip', 'precipcover', 'preciptype', 'windgust',
               'windspeed', 'windir', 'sealevelpressure', 'cloudcover', 'visibility',
               'solarradiation', 'solarenergy', 'icon', 'stations', 'description', 'name', 'winddir_sin', 'winddir_cos']
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors='ignore')

    X_train, X_test, y_train, y_test = get_splits(df, 'conditions')

    # --- BASELINES ---
    # Majority Class Baseline
    majority_class = y_train.mode()[0]
    preds_majority = np.full(shape=y_test.shape, fill_value=majority_class)
    print("Baseline (Majority Class):")
    print(f"  Accuracy: {accuracy_score(y_test, preds_majority):.4f}")
    print(f"  Macro F1: {f1_score(y_test, preds_majority, average='macro'):.4f}\n")

    # Last Observation Baseline (Predict t using true value of t-1)
    preds_last_obs = np.concatenate(([y_train.iloc[-1]], y_test.iloc[:-1]))
    print("Baseline (Last Observation):")
    print(f"  Accuracy: {accuracy_score(y_test, preds_last_obs):.4f}")
    print(f"  Macro F1: {f1_score(y_test, preds_last_obs, average='macro'):.4f}\n")

    # --- MODELING ---
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None]
    }

    rf = RandomForestClassifier(random_state=69, n_jobs=-1)

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=get_tscv(5),
        n_jobs=-1,
        random_state=69,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    print("\nBest params:", search.best_params_)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Model Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")
    print(f"Model Macro F1: {f1_score(y_test, preds, average='macro'):.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=[str(c) for c in le.classes_]))

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')  # Added axis
    plt.ylabel('True Label')       # Added axis
    plt.title("Confusion Matrix: Standard Class (Tuned)")
    plt.tight_layout()
    plt.savefig("S5_Confusion_Standard_Tuned.png")
    print("Saved Confusion Matrix.")



# ==========================================
# 2. NON-PCA REGRESSION
# ==========================================
def run_standard_regression():
    print(f"\n{'=' * 40}\nTASK 2: Standard Regression (No PCA, Tuned)\n{'=' * 40}")
    try:
        df = pd.read_csv("../../S4_lagged_regress.csv")
    except FileNotFoundError:
        print("❌ File 'S4_lagged_regress.csv' not found. Skipping.")
        return

    le = LabelEncoder()
    precip_cols = [c for c in df.columns if c.startswith('preciptype')]
    for col in precip_cols:
        df[col] = le.fit_transform(df[col])

    cond_cols = [c for c in df.columns if c.startswith("conditions")]
    for col in cond_cols:
        df[col] = le.fit_transform(df[col])


    leakage = ['tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike',
               'dew', 'humidity', 'precip', 'precipcover', 'preciptype', 'windgust',
               'windspeed', 'windir', 'sealevelpressure', 'cloudcover', 'visibility',
               'solarradiation', 'solarenergy', 'icon', 'stations', 'description', 'name', 'conditions', 'winddir_sin', 'winddir_cos']
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors='ignore')

    X_train, X_test, y_train, y_test = get_splits(df, 'temp')

    # --- BASELINES ---
    # Last Observation Baseline
    preds_last_obs = np.concatenate(([y_train.iloc[-1]], y_test.iloc[:-1]))
    print("Baseline (Last Observation):")
    print(f"  RMSE: {root_mean_squared_error(y_test, preds_last_obs):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_test, preds_last_obs):.4f}")
    print(f"  R2:  {r2_score(y_test, preds_last_obs):.4f}\n")

    # --- MODELING ---
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None]
    }

    rf = RandomForestRegressor(random_state=69, n_jobs=-1)

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=get_tscv(5),
        n_jobs=-1,
        random_state=69,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    print("Best params:", search.best_params_)
    print(f"Model RMSE: {root_mean_squared_error(y_test, preds):.4f}")
    print(f"Model MAE: {mean_absolute_error(y_test, preds):.4f}")
    print(f"Model R2:  {r2_score(y_test, preds):.4f}")



# ==========================================
# 3. PCA CLASSIFICATION (The Experiment)
# ==========================================
def run_pca_flow():
    print(f"\n{'=' * 40}\nTASK 3: PCA Classification (Tuned)\n{'=' * 40}")
    try:
        df = pd.read_csv("../../S4_lagged_class_no_mi.csv")
    except FileNotFoundError:
        print("❌ File 'S4_lagged_class_no_mi.csv' not found. Skipping.")
        return

    le = LabelEncoder()
    precip_cols = [c for c in df.columns if c.startswith('preciptype')]
    for col in precip_cols:
        df[col] = le.fit_transform(df[col])

    cond_cols = [c for c in df.columns if c.startswith("conditions")]
    for col in cond_cols:
        df[col] = le.fit_transform(df[col])

    leakage = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
               'dew', 'humidity', 'precip', 'precipcover', 'preciptype', 'windgust',
               'windspeed', 'windir', 'sealevelpressure', 'cloudcover', 'visibility',
               'solarradiation', 'solarenergy', 'icon', 'stations', 'description', 'name', 'winddir_sin', 'winddir_cos']
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors='ignore')

    X_train, X_test, y_train, y_test = get_splits(df, 'conditions')

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=69)),
        ("rf", RandomForestClassifier(random_state=69, n_jobs=-1))
    ])

    param_dist = {
        "pca__n_components": [0.80, 0.85, 0.90, 0.95, 0.97, 0.99],
        "rf__n_estimators": [200, 300, 500],
        "rf__max_depth": [10, 15, 25, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 5],
        "rf__max_features": ["sqrt", "log2", None]
    }
 
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=40,
        scoring="f1_macro",
        cv=get_tscv(5),
        n_jobs=-1,
        random_state=69,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    preds = best_pipe.predict(X_test)

    print("Best params:", search.best_params_)
    print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')  # Added axis
    plt.ylabel('True Label')       # Added axis
    plt.title("Confusion Matrix: PCA Class (Tuned)")
    plt.tight_layout()
    plt.savefig("S5_Confusion_PCA_Tuned.png")
    print("Saved Confusion Matrix.")



# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    run_standard_classification()
    run_standard_regression()
    run_pca_flow()


if __name__ == "__main__":
    main()