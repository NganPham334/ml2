import pandas as pd


def validate_merger(s2_path='S2_cleaned.csv', target_path='target.csv'):
    s2 = pd.read_csv(s2_path)
    s2['datetime'] = pd.to_datetime(s2['datetime'])

    target = pd.read_csv(target_path)
    target['datetime'] = pd.to_datetime(target['datetime'])

    # Pick a test index (e.g., the 10th row in your target dataframe)
    test_idx = 10
    test_date = target.iloc[test_idx]['datetime']
    prev_date = test_date - pd.Timedelta(days=1)

    print(f"--- VALIDATING ALIGNMENT FOR {test_date.date()} ---")

    # 1. Check Day t (Ground Truth Target)
    expected_target_temp = s2[s2['datetime'] == test_date]['temp'].values[0]
    actual_target_temp = target.iloc[test_idx]['target_temp']

    print(f"Target Temp (Day t): Expected={expected_target_temp}, Actual={actual_target_temp}")
    if expected_target_temp == actual_target_temp:
        print("✅ Target variable aligned correctly (No data leakage from future days).")
    else:
        print("❌ Mismatch in target variable!")

    # 2. Check Day t-1 (Lag Feature)
    if 'temp_lag' in target.columns:
        expected_lag_temp = s2[s2['datetime'] == prev_date]['temp'].values[0]
        actual_lag_temp = target.iloc[test_idx]['temp_lag']

        print(f"\nLag Temp (Day t-1): Expected={expected_lag_temp}, Actual={actual_lag_temp}")
        if expected_lag_temp == actual_lag_temp:
            print("✅ Lag feature aligned correctly (Properly pulling from yesterday).")
        else:
            print("❌ Mismatch in lag feature!")

    # 3. Quick visual check of the row
    print("\nVisual Check of Target Row:")
    display_cols = ['datetime', 'target_temp', 'target_conditions', 'temp_lag', 'conditions_lag']
    # Filter to only columns that actually exist to avoid KeyError
    display_cols = [c for c in display_cols if c in target.columns]
    print(target.iloc[test_idx:test_idx + 1][display_cols].to_string(index=False))


# Run the validation
validate_merger()