import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, mean_absolute_error, mean_squared_error, r2_score, \
    confusion_matrix, accuracy_score
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
    df['conditions'] = le.fit_transform(df['conditions'])
    cond_feat_cols = [c for c in df.columns if c.startswith("conditions_")]

    for col in cond_feat_cols:
        df[col] = (
            df[col]
            .astype(str)
            .map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )

    leakage = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
               'dew', 'humidity', 'precip', 'precipcover', 'preciptype', 'windgust',
               'windspeed', 'windir', 'sealevelpressure', 'cloudcover', 'visibility',
               'solarradiation', 'solarenergy', 'icon', 'stations', 'description', 'name', 'windir_sin', 'windir_cos']
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors='ignore')

    precip_cols = [c for c in df.columns if c.startswith('preciptype')]
    for col in precip_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X_train, X_test, y_train, y_test = get_splits(df, 'conditions')

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

    print("Best params:", search.best_params_)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")
    print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix: Standard Class (Tuned)")
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

    precip_cols = [c for c in df.columns if c.startswith('preciptype')]
    for col in precip_cols:
        df[col] = df[col].astype(str).fillna('none')
        df[col] = LabelEncoder().fit_transform(df[col])

    le = LabelEncoder()
    df['conditions'] = le.fit_transform(df['conditions'])
    cond_feat_cols = [c for c in df.columns if c.startswith("conditions_")]

    for col in cond_feat_cols:
        df[col] = (
            df[col]
            .astype(str)
            .map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )

    leakage = ['tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike',
               'dew', 'humidity', 'precip', 'precipcover', 'preciptype', 'windgust',
               'windspeed', 'windir', 'sealevelpressure', 'cloudcover', 'visibility',
               'solarradiation', 'solarenergy', 'icon', 'stations', 'description', 'name', 'conditions', 'windir_sin', 'windir_cos']
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors='ignore')

    X_train, X_test, y_train, y_test = get_splits(df, 'temp')

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
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2:  {r2_score(y_test, preds):.4f}")



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
    df['conditions'] = le.fit_transform(df['conditions'])
    cond_feat_cols = [c for c in df.columns if c.startswith("conditions_")]

    for col in cond_feat_cols:
        df[col] = (
            df[col]
            .astype(str)
            .map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )

    leakage = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
               'dew', 'humidity', 'precip', 'precipcover', 'preciptype', 'windgust',
               'windspeed', 'windir', 'sealevelpressure', 'cloudcover', 'visibility',
               'solarradiation', 'solarenergy', 'icon', 'stations', 'description', 'name', 'windir_sin', 'windir_cos']
    df = df.drop(columns=[c for c in leakage if c in df.columns], errors='ignore')

    precip_cols = [c for c in df.columns if c.startswith('preciptype')]
    for col in precip_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

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
    plt.title("Confusion Matrix: PCA Class (Tuned)")
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