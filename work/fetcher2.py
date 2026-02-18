import pandas as pd
import requests
import time
import sys

# --- CONFIGURATION ---
API_KEY = "2VK8ABRWJRPB3N37EJDPQ87WU"
LOCATION = "Cau Giay,Hanoi,VN"
START_DATE = "2020-01-02"
END_DATE = "2025-06-21"
OUTPUT_FILE = "VC_Exact_Hours_Full.csv"

# The specific hours you want to query for each calendar day
TARGET_HOURS = ['00:00:00', '22:00:00', '23:00:00']

# --- RATE LIMITING ---
# 1.5 seconds = ~40 requests per minute.
# If the script halts with a 429 error, increase this to 2.0 or 3.0.
REQUEST_DELAY_SECONDS = 0.8


def fetch_specific_hours():
    print(f"Generating exact daily schedule from {START_DATE} to {END_DATE}...")

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    all_records = []

    total_requests = len(dates) * len(TARGET_HOURS)
    current_request = 0

    print(f"Total days: {len(dates)} | Target Hours per day: {len(TARGET_HOURS)}")
    print(f"Total API calls scheduled: {total_requests}")
    print("-" * 50)

    # Use enumerate to keep track of exactly how many days we have processed
    for day_index, single_date in enumerate(dates):
        date_str = single_date.strftime('%Y-%m-%d')

        for hour_str in TARGET_HOURS:
            current_request += 1
            datetime_param = f"{date_str}T{hour_str}"

            # Print live progress on the SAME line (using \r and flush=True)
            print(f"\rProgress: {current_request}/{total_requests} requests completed | Fetching {datetime_param}...",
                  end="", flush=True)

            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}/{datetime_param}?unitGroup=metric&key={API_KEY}&include=current&contentType=json"

            try:
                response = requests.get(url)

                # Defensively catch API Limits
                if response.status_code == 429:
                    print(f"\n\n🚨 ERROR: API Limit or Rate Limit Reached at {datetime_param}!")
                    print(f"Try increasing REQUEST_DELAY_SECONDS (currently {REQUEST_DELAY_SECONDS}s).")
                    save_data(all_records, silent=False)
                    sys.exit()

                elif response.status_code != 200:
                    print(f"\n⚠️ Warning: Failed on {datetime_param}. Status: {response.status_code}. Skipping...")
                    continue

                data = response.json()

                if 'currentConditions' in data:
                    current = data['currentConditions']

                    # --- NEW LOGIC: Save EVERYTHING ---
                    # We inject our custom tracking timestamp directly into the VC dictionary
                    current['query_datetime'] = datetime_param

                    # We append the ENTIRE dictionary instead of mapping specific keys
                    all_records.append(current)

            except Exception as e:
                print(f"\n❌ Crash prevented on {datetime_param}: {e}")

            # Pause to respect API rate limits
            time.sleep(REQUEST_DELAY_SECONDS)

        # Checkpoint Save Every 3 Days
        if (day_index + 1) % 3 == 0:
            save_data(all_records, silent=True)

    # Move to a new line after the progress tracker finishes
    print("\n\nData extraction complete!")
    save_data(all_records, silent=False)


def save_data(records_list, silent=False):
    """Helper function to save cleanly. Uses 'silent' to avoid messing up terminal progress bars."""
    if len(records_list) > 0:
        df = pd.DataFrame(records_list)
        df.to_csv(OUTPUT_FILE, index=False)
        if not silent:
            print(f"💾 Saved {len(df)} specific hour records to '{OUTPUT_FILE}'")
    else:
        if not silent:
            print("⚠️ No records to save.")


if __name__ == "__main__":
    fetch_specific_hours()