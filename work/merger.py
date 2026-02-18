import pandas as pd
import os

# 1. Load the two files
# (Assuming they are in the same folder as your script)
df1 = pd.read_csv('orig_dataset/cầu giấy 2020-01-01 to 2022-09-26.csv')
df2 = pd.read_csv('orig_dataset/cầu giấy 2022-09-27 to 2025-06-21.csv')
output = "S1_merged.csv"

# 2. The Safety Check (Crucial!)
# This code checks if the column names are IDENTICAL
if list(df1.columns) == list(df2.columns):
    print("✅ Success! Columns match perfectly.")

    # 3. Combine them
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 4. Save the new "Master File"
    combined_df.to_csv(output, index=False)
    print(f"Saved {output} with {len(combined_df)} rows.")

else:
    print("❌ Error! Columns do not match.")
    # Show the difference so you can fix it
    print("Columns in File 1 but not File 2:", set(df1.columns) - set(df2.columns))
    print("Columns in File 2 but not File 1:", set(df2.columns) - set(df1.columns))