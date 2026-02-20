import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def prep_features(df):
    # Strictly prevent target leakage by dropping both targets and the datetime identifier
    X = df.drop(columns=['datetime', 'target_temp', 'target_conditions'], errors='ignore')
    # One-hot encode any remaining text columns (e.g., categorical lag features)
    X = pd.get_dummies(X, drop_first=True)
    return X


def rf_classification(df):
    X = prep_features(df)
    y = df['target_conditions']

    # Drop rows where target is NaN just in case
    valid_idx = y.dropna().index
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_grid, n_iter=40, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    print("--- Classification Results ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")
    print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Generate and save Permutation Importance
    result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    imp_df.to_csv('rf_classification_importance.csv', index=False)
    print("\nSaved classification feature importance to rf_classification_importance.csv\n")

    return best_model


def rf_regression(df):
    X = prep_features(df)
    y = df['target_temp']

    valid_idx = y.dropna().index
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 1.0]
    }

    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(rf, param_grid, n_iter=40, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1,
                                random_state=42)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    print("--- Regression Results ---")
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
    print(f"R-squared: {r2_score(y_test, preds):.4f}\n")

    return best_model


if __name__ == "__main__":
    df = pd.read_csv('target.csv')

    print("Starting Classification Tuning...")
    clf_model = rf_classification(df)

    print("Starting Regression Tuning...")
    reg_model = rf_regression(df)