import pandas as pd
from scipy.signal import resample_poly

# Load the original CSV file
original_df = pd.read_csv('Matthew_Perturb_ChestBackShrugRightLeft.csv')

# Assuming the CSV has a column named 'time' for timestamps and a column named 'data' for the data values
time_column = 'Unit'
data_column = 'V'

# Set the sampling rates
original_sampling_rate = 125000  # Hz
target_sampling_rate = 1000  # Hz

# Calculate the resampling ratio
resampling_ratio = original_sampling_rate // target_sampling_rate

# Resample the data
resampled_data = resample_poly(original_df[data_column], target_sampling_rate, original_sampling_rate)

# Resample the time stamps
resampled_time = original_df[time_column][::resampling_ratio]

# Create a new DataFrame with resampled data and timestamps
resampled_df = pd.DataFrame({time_column: resampled_time, data_column: resampled_data})

# Save the resampled data to a new CSV file
resampled_df.to_csv('resampled_Matthew_Perturb_ChestBackShrugRightLeft.csv', index=False)
