import os
import random
import csv
import pandas as pd


def remove_extra_columns(data, num_desired_columns):
    if len(data.columns) <= num_desired_columns:
    # No extra columns to remove
        return data
    # Assuming extra columns are at the end
    return data.iloc[:, :num_desired_columns]

def combine_and_shuffle_csv(data_dir, output_file):
    all_data = []
    # Loop through each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):  # Check for CSV files only
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                # Skip header row (if present)
                next(reader, None)  # Assuming the first row is the header
                data = list(reader)  # Read all remaining rows
                all_data.extend(data)  # Add data to the combined list
    return all_data

# Example usage
data_dir = 'data_in_csv'  # Replace with your data directory path
output_file = 'combined_shuffled_data.csv'  # Name of the combined output file
data = combine_and_shuffle_csv(data_dir, output_file)
cleaned_df = []
for row in data:
    d = []
    ct = 0
    for cell in row:
        if(ct > 42):
            break
        d.append(cell)
        ct+=1
    cleaned_df.append(d)
    
random.shuffle(cleaned_df)

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    header = []
    header.append('Class')
    for i in range(21):
        header.append(f'x{i}')
        header.append(f'y{i}')
    writer.writerow(header)
    # Write all rows to the output file
    writer.writerows(cleaned_df)

print(f"Data combined and shuffled, saved to {output_file}")
