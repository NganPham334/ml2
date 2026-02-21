import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

input_file = "S2_cleaned.csv"


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def print_raw_audit(df):
    if 'conditions' in df.columns:
        print("\n--- RAW CONDITION OCCURRENCES ---")
        print(df['conditions'].value_counts().to_string())
        print("-" * 30)


def encode_cyclical(df, col, max_val):
    if col in df.columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        df = df.drop(columns=[col])
    return df


def prep_common_features(df):
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

    # --- NEW TIME-SERIES METRIC FEATURES (STRICTLY HISTORICAL) ---
    # 1. Rolling 3-day precipitation sum (ends yesterday)
    if 'precip' in df.columns:
        df['rolling_precip_3day'] = df['precip'].shift(1).rolling(window=3).sum()

    # 2. Diurnal temperature range (from yesterday)
    if 'tempmax' in df.columns and 'tempmin' in df.columns:
        df['diurnal_temp_range'] = (df['tempmax'] - df['tempmin']).shift(1)

    # 3. 24h Pressure change (change between yesterday and the day before yesterday)
    if 'sealevelpressure' in df.columns:
        df['pressure_change_24h'] = df['sealevelpressure'].diff(1).shift(1)
    elif 'pressure' in df.columns:
        df['pressure_change_24h'] = df['pressure'].diff(1).shift(1)
    # -------------------------------------------------------------

    if 'datetime' in df.columns:
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df = encode_cyclical(df, 'day_of_year', 365.25)

    df = encode_cyclical(df, 'winddir', 360.0)

    if 'conditions' in df.columns:
        df.loc[df['conditions'].str.contains('Rain', case=False, na=False), 'conditions'] = 'Rain'
        df.loc[df['conditions'].str.contains('cloudy|Overcast', case=False, na=False), 'conditions'] = 'Cloudy'

    return df


def add_lags(df):
    """Lags features 1-3 days, excluding specific columns."""
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

    # STRICT LOGIC REQUESTED - Excluded new engineered features
    exclude_cols = [
        'datetime',
        'day_of_year_sin',
        'day_of_year_cos',
        'rolling_precip_3day',
        'diurnal_temp_range',
        'pressure_change_24h'
    ]
    cols_to_lag = [col for col in df.columns if col not in exclude_cols]

    for col in cols_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"   -> Lagged {len(cols_to_lag)} features. Dropped {initial_len - len(df)} rows.")
    return df


# ==========================================
# 2. FEATURE SELECTION (MI)
# ==========================================
def drop_correlated_features_by_mi(df, target_col, task_type='classification', threshold=0.90):
    print(f"\n--- Feature Selection: '{target_col}' ({task_type}) ---")
    if target_col not in df.columns: return df

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    math_df = df[numeric_cols].copy().fillna(df[numeric_cols].median())

    corr_matrix = math_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = []
    for c in upper.columns:
        for r in upper.index:
            if upper.loc[r, c] > threshold: high_corr_pairs.append((r, c, upper.loc[r, c]))

    if not high_corr_pairs:
        print("   No highly correlated features found.")
        return df

    mi_mask = df[target_col].notna()
    if task_type == 'classification':
        le = LabelEncoder()
        y_temp = df.loc[mi_mask, target_col]
        if y_temp.dtype == 'object': y_temp = le.fit_transform(y_temp)
        mi_scores = mutual_info_classif(math_df[mi_mask], y_temp, random_state=69)
    else:
        mi_scores = mutual_info_regression(math_df[mi_mask], df.loc[mi_mask, target_col], random_state=69)

    mi_dict = dict(zip(numeric_cols, mi_scores))
    cols_to_drop = set()

    for f1, f2, _ in high_corr_pairs:
        if f1 in cols_to_drop or f2 in cols_to_drop: continue
        drop_col = f1 if mi_dict.get(f1, 0) < mi_dict.get(f2, 0) else f2
        cols_to_drop.add(drop_col)

    if cols_to_drop:
        df = df.drop(columns=list(cols_to_drop))
        print(f"🗑️ DROPPED {len(cols_to_drop)} features: {', '.join(sorted(cols_to_drop))}")

    return df


# ==========================================
# 3. PCA LOGIC
# ==========================================
# def apply_pca(df, target_col='conditions', variance_to_keep=0.95):
#     print(f"\n--- Running PCA ---")
#     meta_cols = ['datetime', target_col]
#     df_meta = df[meta_cols].copy()
#     feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
#
#     X_total = df[feature_cols].copy().fillna(df[feature_cols].median())
#
#     # Strict Split
#     split_idx = int(len(X_total) * 0.8)
#     X_train = X_total.iloc[:split_idx]
#
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_scaled_total = scaler.transform(X_total)
#
#     pca = PCA(n_components=variance_to_keep, random_state=69)
#     pca.fit(X_scaled_total[:split_idx])  # Fit only on train
#
#     X_pca = pca.transform(X_scaled_total)
#     print(f"   -> Compressed {len(feature_cols)} features into {X_pca.shape[1]} PCs.")
#
#     pc_cols = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
#     df_pca = pd.DataFrame(X_pca, columns=pc_cols)
#     return pd.concat([df_meta.reset_index(drop=True), df_pca], axis=1)


# ==========================================
# MAIN
# ==========================================
def main():
    try:
        df_base = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found.")
        return

    print_raw_audit(df_base)
    df_base = prep_common_features(df_base)

    # 1. Standard Class (Selection -> Lag)
    df_class = drop_correlated_features_by_mi(df_base.copy(), 'conditions', 'classification')
    df_class = add_lags(df_class)
    df_class.to_csv("S4_lagged_class.csv", index=False)
    print("✅ Saved S4_lagged_class.csv")

    # 2. Standard Regress (Selection -> Lag)
    df_regress = drop_correlated_features_by_mi(df_base.copy(), 'temp', 'regression')
    df_regress = add_lags(df_regress)
    df_regress.to_csv("S4_lagged_regress.csv", index=False)
    print("✅ Saved S4_lagged_regress.csv")

    # 3. PCA Flow (Only lag, we will apply PCA in the model pipeline later)
    df_pca = add_lags(df_base.copy())
    df_pca.to_csv("S4_lagged_class_no_mi.csv", index=False)
    print("✅ Saved S4_lagged_class_no_mi.csv")


if __name__ == "__main__":
    main()