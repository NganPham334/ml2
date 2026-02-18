import pandas as pd

# 1. Define your input and output file names
input_file = "S1_merged.csv"
output_file = "S2_cleaned.csv"

# 2. List the exact names of the columns you want to discard
columns_to_discard = [
    "name",
    "stations",
    "snow",
    "snowdepth",
    "severerisk",
    "sunrise",
    "sunset",
    "description",
    "icon",
    "uvindex",
    "precipprob"
]


def clean_data():
    try:
        # Load the dataset
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)

        # Check which columns actually exist in the dataframe to avoid errors
        existing_cols_to_drop = [col for col in columns_to_discard if col in df.columns]

        # Drop the columns
        df_cleaned = df.drop(columns=existing_cols_to_drop)

        if 'preciptype' in df_cleaned.columns:
            df_cleaned['preciptype'] = df_cleaned['preciptype'].fillna('noprecip')
            print("Filled missing 'preciptype' values with 'noprecip'.")

        # Drop zero-variance features (theres none, we're doing this as a formality since its mentioned in the slides)
        zero_var_cols = df_cleaned.columns[df_cleaned.nunique() <= 1].tolist()

        if zero_var_cols:
            df_cleaned = df_cleaned.drop(columns=zero_var_cols)
            print(f"Dropped zero-variance features: {zero_var_cols}")

        # Save the cleaned dataset to a new CSV
        df_cleaned.to_csv(output_file, index=False)

        print(f"\nSuccess! Removed {len(existing_cols_to_drop)} columns.")
        print(f"Cleaned data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the spelling and location.")


if __name__ == "__main__":
    clean_data()