import pandas as pd

# Process all 3 datasets
files_to_process = {
    "S3_engineered_class.csv": "S4_lagged_class.csv",
    "S3_engineered_regress.csv": "S4_lagged_regress.csv",
    "S3_engineered_class_full.csv": "S4_lagged_class_full.csv" # New PCA Flow
}

def create_dynamic_lag_features(input_file, output_file):
    print(f"\nProcessing {input_file}...")
    df = pd.read_csv(input_file)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

    exclude_cols = ['datetime', 'conditions', 'day_of_year_sin', 'day_of_year_cos', 'moonphase']
    cols_to_lag = [col for col in df.columns if col not in exclude_cols and col != 'datetime']

    for col in cols_to_lag:
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    df_clean = df.dropna().reset_index(drop=True)
    df_clean.to_csv(output_file, index=False)
    print(f"✅ Success! Saved {len(cols_to_lag)*3} lag features to {output_file}")

if __name__ == "__main__":
    for in_file, out_file in files_to_process.items():
        try:
            create_dynamic_lag_features(in_file, out_file)
        except FileNotFoundError:
            print(f"❌ Error: '{in_file}' not found.")