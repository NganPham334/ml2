import pandas as pd
import numpy as np


def clean_numerical(df, name):
    numeric_cols = df.select_dtypes(include=['number']).columns
    cols_to_fix = [col for col in numeric_cols if df[col].isnull().any()]
    if cols_to_fix:
        df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def before_merge_treatment(df, name):
    if 'winddir' in df.columns:
        radians = df['winddir'] * np.pi / 180
        df['winddir_sin'] = np.sin(radians)
        df['winddir_cos'] = np.cos(radians)

    if 'conditions' in df.columns:
        def consolidate(val):
            if pd.isna(val): return val
            low_val = str(val).lower()
            if 'rain' in low_val: return 'Rain'
            if 'overcast' in low_val or 'partially cloudy' in low_val: return 'Cloudy'
            if 'clear' in low_val: return 'Clear'
            return val

        df['conditions'] = df['conditions'].apply(consolidate)

    if name == "VC_Exact" and 'preciptype' in df.columns:
        df['preciptype'] = df['preciptype'].fillna('noprecip')
        df['preciptype'] = df['preciptype'].astype(str).str.replace(r"[\[\]']", "", regex=True)

    if name == "S2_Cleaned" and 'precip' in df.columns:
        df['rolling_precip_3day'] = df['precip'].rolling(window=3).sum()

    return df


def after_merge_treatment(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    day_of_year = df['datetime'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    def get_angular_diff(dir1, dir2):
        return ((dir1 - dir2 + 180) % 360 - 180).abs()

    df['winddir_diff_2h'] = get_angular_diff(df['winddir_lag_12AM'], df['winddir_lag_10PM'])
    df['winddir_diff_day'] = get_angular_diff(df['winddir_lag_12AM'], df['winddir_lag'])

    df['cloud_cover_trend_2h'] = df['cloudcover_lag_12AM'] - df['cloudcover_lag_10PM']

    if 'tempmax_lag' in df.columns and 'tempmin_lag' in df.columns:
        df['diurnal_temp_range'] = df['tempmax_lag'] - df['tempmin_lag']

    df['dewpoint_depression'] = df['temp_lag_12AM'] - df['dew_lag_12AM']
    df['humidity_momentum'] = df['humidity_lag_12AM'] - df['humidity_lag_10PM']
    df['pressure_trend_overnight'] = df['pressure_lag_12AM'] - df['pressure_lag_10PM']

    if 'rolling_precip_3day_lag' in df.columns:
        df = df.rename(columns={'rolling_precip_3day_lag': 'rolling_precip_3day'})

    cols_to_drop = [
        'winddir_lag_12AM', 'winddir_lag_10PM', 'winddir_lag',
        'cloudcover_lag_10PM', 'humidity_lag_10PM', 'pressure_lag_10PM'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def merge_and_prepare(s2_path, vc_path, output_path):
    s2 = pd.read_csv(s2_path)
    vc = pd.read_csv(vc_path)

    if 'datetime' in s2.columns:
        s2['datetime'] = pd.to_datetime(s2['datetime'])
        s2 = s2.sort_values('datetime').reset_index(drop=True)

    if 'query_datetime' in vc.columns:
        vc['query_datetime'] = pd.to_datetime(vc['query_datetime'])
        vc = vc.sort_values('query_datetime').reset_index(drop=True)

    s2 = clean_numerical(s2, "S2_Cleaned")
    vc = clean_numerical(vc, "VC_Exact")

    s2 = before_merge_treatment(s2, "S2_Cleaned")
    vc = before_merge_treatment(vc, "VC_Exact")

    vc_exclude = [
        'datetimeEpoch', 'snowdepth', 'solarradiation', 'solarenergy',
        'uvindex', 'icon', 'stations', 'source', 'sunrise',
        'sunriseEpoch', 'sunset', 'sunsetEpoch', 'moonphase', 'severerisk', 'datetime',
        'precipprob'
    ]
    s2_exclude = ['moonphase']

    vc_12am = vc[vc['query_datetime'].dt.hour == 0].copy()
    vc_12am['join_date'] = vc_12am['query_datetime'].dt.normalize()
    vc_cols = [c for c in vc_12am.columns if c not in vc_exclude + ['query_datetime', 'join_date']]
    vc_subset = vc_12am[['join_date'] + vc_cols].rename(columns={c: f"{c}_lag_12AM" for c in vc_cols})

    vc_10pm = vc[vc['query_datetime'].dt.hour == 22].copy()
    vc_10pm['join_date_shifted'] = vc_10pm['query_datetime'].dt.normalize() + pd.Timedelta(days=1)
    vc_10pm_subset = vc_10pm[['join_date_shifted', 'winddir', 'cloudcover', 'humidity', 'pressure']].rename(
        columns={
            'winddir': 'winddir_lag_10PM',
            'cloudcover': 'cloudcover_lag_10PM',
            'humidity': 'humidity_lag_10PM',
            'pressure': 'pressure_lag_10PM'
        }
    )

    s2_lag = s2.copy()
    s2_lag['join_date_shifted'] = s2_lag['datetime'] + pd.Timedelta(days=1)
    s2_cols = [c for c in s2_lag.columns if c not in s2_exclude + ['datetime', 'join_date_shifted']]
    s2_subset = s2_lag[['join_date_shifted'] + s2_cols].rename(columns={c: f"{c}_lag" for c in s2_cols})

    # --- UPDATED SECTION ---
    # Extract the target variables for Day t and rename them
    target_cols = ['datetime']
    if 'temp' in s2.columns: target_cols.append('temp')
    if 'conditions' in s2.columns: target_cols.append('conditions')

    target = s2.iloc[7:].copy()[target_cols]
    target = target.rename(columns={'temp': 'target_temp', 'conditions': 'target_conditions'})
    # -----------------------

    target = pd.merge(target, vc_subset, left_on='datetime', right_on='join_date', how='left')
    target = target.drop(columns=['join_date'])

    target = pd.merge(target, vc_10pm_subset, left_on='datetime', right_on='join_date_shifted', how='left')
    target = target.drop(columns=['join_date_shifted'])

    target = pd.merge(target, s2_subset, left_on='datetime', right_on='join_date_shifted', how='left')
    target = target.drop(columns=['join_date_shifted'])

    target = after_merge_treatment(target)

    target.to_csv(output_path, index=False)
    print(f"Successfully saved {output_path} with target_temp and target_conditions included.")


if __name__ == "__main__":
    merge_and_prepare('S2_cleaned.csv', 'VC_Exact_Hours_Full.csv', 'target.csv')