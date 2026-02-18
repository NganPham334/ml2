import requests
from datetime import datetime, timedelta

# Configuration
API_KEY = "2VK8ABRWJRPB3N37EJDPQ87WU"  # Replace with your actual Visual Crossing API key
LOCATION = "Hanoi,VN"


def fetch_yesterday_9pm():
    # 1. Calculate exactly yesterday's date
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    # 2. Hardcode the time to 9:00 PM (21:00:00)
    target_time = yesterday.replace(hour=21, minute=0, second=0, microsecond=0)

    # 3. Format exactly as Visual Crossing demands: YYYY-MM-DDTHH:00:00
    vc_time_string = target_time.strftime("%Y-%m-%dT%H:%M:%S")

    # 4. Construct the URL.
    # Notice &include=current -> This is the magic parameter that prevents a 24-credit charge.
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{LOCATION}/{vc_time_string}?key={API_KEY}&include=current&contentType=json"
    )

    print(f"Requesting historical data for: {vc_time_string}")
    print("Sending API request...\n")

    # 5. Make the request
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # 6. Extract the API's internal receipt
        query_cost = data.get('queryCost', 'Unknown')

        # Even though it is historical data, it populates in 'currentConditions'
        # because we passed the specific hour into the URL path!
        if 'currentConditions' in data:
            temp = data['currentConditions'].get('temp')
            conditions = data['currentConditions'].get('conditions')

            print(f"✅ SUCCESS!")
            print(f"Location: {data.get('resolvedAddress')}")
            print(f"Timestamp: {vc_time_string}")
            print(f"Temperature: {temp}°")
            print(f"Conditions: {conditions}")
            print("-" * 40)

            # THE PROOF
            print(f"💰 ACTUAL CREDIT COST: {query_cost}")
            if query_cost == 1:
                print(
                    "Proof Confirmed: The &include=current parameter successfully restricted the charge to exactly 1 credit!")
            else:
                print("Warning: The query cost was higher than expected.")
        else:
            print("Error: 'currentConditions' not found in the response.")

    elif response.status_code == 401:
        print("🚨 Error 401: Unauthorized. Please check your API_KEY.")
    elif response.status_code == 429:
        print("🚨 Error 429: You have exceeded your daily 1,000 credit limit.")
    else:
        print(f"❌ Failed to retrieve data. HTTP Status code: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    fetch_yesterday_9pm()