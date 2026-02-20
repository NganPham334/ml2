import pandas as pd
import numpy as np
import datetime

# --- Configuration ---
S2_FILE = "S2_cleaned.csv"
VC_FILE = "VC_Exact_Hours_Full.csv"
OUTPUT_FILE = "processed_weather_data.csv"

def load_and_clean_s2():
    print("Loading S2_cleaned.csv...")
    df = pd.read_csv(S2_FILE)
    
    # Ensure datetime conversion
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by date
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df

def load_and_clean_vc():
    print("Loading VC_Exact_Hours_Full.csv...")
    # Header inference based on fetcher2.py and inspection
    # It seems to dump the JSON response 'currentConditions' dict keys or similar.
    # Looking at the head output from earlier:
    # 00:00:00,1577898000,20.0,20.0,88.3,18.0,0.0,0.0,0.0,0.0,,,5.4,330.0,1023.0,3.0,69.9,0.0,0.0,0.0,Partially cloudy,partly-cloudy-night,"['48820099999', 'VVNB']",obs,06:34:04,1577921644,17:26:56,1577960816,0.22,2020-01-02T00:00:00,
    
    # Based on standard VC API response text/csv format (usually):
    # name, datetime, temp, feelslike, dew, humidity, precip, ...
    # But here we have specific columns. Let's map them based on values.
    # 0: datetime (time only "00:00:00")
    # 1: datetimeEpoch
    # 2: temp
    # 3: feelslike
    # 4: humidity
    # 5: dew
    # 6: precip
    # 7: precipProb
    # 8: snow
    # 9: snowDepth
    # 10: precipType
    # 11: windGust
    # 12: windSpeed
    # 13: windDir
    # 14: pressure
    # 15: visibility
    # 16: cloudCover
    # 17: solarRadiation
    # 18: solarEnergy
    # 19: uvIndex
    # 20: conditions
    # 21: icon
    # 22: stations
    # 23: source
    # 24: sunrise
    # 25: sunriseEpoch
    # 26: sunset
    # 27: sunsetEpoch
    # 28: moonphase
    # 29: query_datetime (added by fetcher2.py)
    
    # Wait, the head output has "20.0,20.0" (temp, feelslike?) then "88.3" (humidity?), "18.0" (dew?).
    # Let's verify standard VC CSV or JSON flat structure.
    # fetcher2.py appends `current` dict. 
    # `pd.DataFrame(records_list)` uses keys as columns. 
    # If the file has NO header, it means it was saved in append mode or without header initially?
    # Actually fetcher2.py `to_csv(..., index=False)` keeps header by default. 
    # But the user said "missing header". 
    # If fetcher2.py was run multiple times or modified, maybe headers are missing or repeated.
    
    # Let's assume the columns based on the sample row provided in context:
    # 0: time ("00:00:00")
    # 1: datetimeEpoch
    # 2: temp
    # 3: feelslike (?) - wait, 20.0 and 20.0.
    # 4: humidity (88.3) - plausibly humidity (high at night)
    # 5: dew (18.0) - dew point < temp
    # ...
    # 29: query_datetime is definitely the last one: "2020-01-02T00:00:00"
    
    # We really only need a few specific columns.
    # query_datetime is key because it contains the full date and time we queried.
    
    try:
        df = pd.read_csv(VC_FILE, header=None)
    except Exception as e:
        print(f"Error reading VC file: {e}")
        return None

    # Rename critical columns
    # We will rely on index 29 (last one) being the full iso timestamp
    # 2=temp, 4=humidity, 20=conditions, 14=pressure
    # Let's hope these indices are stable.
    
    # If there are varying number of columns, this might break.
    # The last column is query_datetime.
    
    df.rename(columns={
        29: 'query_datetime',
        2: 'hour_temp',
        4: 'hour_humidity',
        14: 'hour_pressure',
        20: 'hour_conditions'
    }, inplace=True)
    
    # Convert query_datetime to datetime objects
    df['query_datetime'] = pd.to_datetime(df['query_datetime'])
    
    # Create separate Date and Hour columns for joining
    df['date'] = df['query_datetime'].dt.date
    df['hour'] = df['query_datetime'].dt.hour
    
    # Filter for relevant hours: 22 (10PM), 23 (11PM), 0 (12AM)
    # Note: 00:00:00 belongs to the START of the day.
    # If we want 12AM of day T, that is hour 0 of day T.
    # If we want 10PM, 11PM of previous day, we need those relative to day T.
    
    return df[['date', 'hour', 'hour_temp', 'hour_humidity', 'hour_pressure', 'hour_conditions']]

def simplify_conditions(cond):
    if pd.isna(cond):
        return "Cloudy" # Default assumption or drop
    
    cond_lower = str(cond).lower()
    
    if 'rain' in cond_lower:
        return 'Rain'
    elif 'clear' in cond_lower or 'sunny' in cond_lower:
        return 'Clear'
    else:
        return 'Cloudy' # Covers 'Partially cloudy', 'Overcast', etc.

def prepare_dataset():
    df_s2 = load_and_clean_s2()
    df_vc = load_and_clean_vc()
    
    print("Simplifying target conditions...")
    df_s2['target_conditions'] = df_s2['conditions'].apply(simplify_conditions)
    
    # Feature Engineering
    print("Engineering features...")
    
    final_rows = []
    
    # We can't predict for the very first few days because we need lags
    LAG_DAYS = 3
    
    # Iterate through days in S2
    # We treat each row in S2 as day T (the day we want to predict)
    # We use T's date info + moonphase
    # We use lag features from T-1, T-2...
    # We use hourly data from T-1 (22:00, 23:00) and T (00:00)
    
    start_idx = LAG_DAYS
    
    for i in range(start_idx, len(df_s2)):
        day_t = df_s2.iloc[i]
        date_t = day_t['datetime'].date()
        
        row = {
            'date': date_t,
            'target': day_t['target_conditions'],
            # Feature: Moonphase (current day)
            'moonphase': day_t['moonphase'],
            # Feature: Day of Year
            'day_of_year': day_t['datetime'].dayofyear
        }
        
        # Lag features from S2 (Daily summaries)
        for lag in range(1, LAG_DAYS + 1):
            prev_day = df_s2.iloc[i - lag]
            prefix = f"lag{lag}_"
            row[prefix + 'temp'] = prev_day['temp']
            row[prefix + 'humidity'] = prev_day['humidity']
            row[prefix + 'precip'] = prev_day['precip']
            row[prefix + 'windspeed'] = prev_day['windspeed']
            # We could also encode previous conditions, but let's stick to numeric for now or simple encode
            row[prefix + 'conditions_simple'] = simplify_conditions(prev_day['conditions'])
            
        # Hourly features from VC
        # Need 22:00, 23:00 of Day T-1
        date_prev = date_t - datetime.timedelta(days=1)
        
        # Helper to find hourly data
        def get_hour_data(d, h, suffix):
            match = df_vc[(df_vc['date'] == d) & (df_vc['hour'] == h)]
            if not match.empty:
                r = match.iloc[0]
                row[f'h_{suffix}_temp'] = r['hour_temp']
                row[f'h_{suffix}_humidity'] = r['hour_humidity']
                row[f'h_{suffix}_pressure'] = r['hour_pressure']
                row[f'h_{suffix}_cond'] = simplify_conditions(r['hour_conditions'])
            else:
                # Handle missing data - forward fill or NaN? 
                # For now let's leave NaN and handle later or fill with Daily avg
                row[f'h_{suffix}_temp'] = np.nan
                row[f'h_{suffix}_humidity'] = np.nan
                row[f'h_{suffix}_pressure'] = np.nan
                row[f'h_{suffix}_cond'] = 'Unknown'

        get_hour_data(date_prev, 22, '10pm_prev')
        get_hour_data(date_prev, 23, '11pm_prev')
        get_hour_data(date_t, 0, '12am_curr') # 00:00 of day T
        
        final_rows.append(row)
        
    final_df = pd.DataFrame(final_rows)
    
    # Drop rows with NaN if any (or fill)
    # Using dropna for strict quality first
    initial_len = len(final_df)
    final_df = final_df.dropna()
    print(f"Created dataset with {len(final_df)} rows (dropped {initial_len - len(final_df)} due to missing data).")
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dataset()
