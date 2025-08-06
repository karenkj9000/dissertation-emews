import os
import numpy as np
import pandas as pd


def upsample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Find the split point between 24hr and 12hr data
    df['time_diff'] = df.index.to_series().diff()

    # Find the first index where the interval becomes 12 hours
    twelve_hour_intervals = df[df['time_diff'] == pd.Timedelta(hours=12)]

    if twelve_hour_intervals.empty:
        print("No 12-hour interval found. Returning original data.")
        df.drop(columns=['time_diff'], inplace=True)
        return df

    first_12hr_idx = twelve_hour_intervals.index[0] - pd.Timedelta(hours=12)

    # Split the DataFrame into two segments
    df_24hr_segment = df.loc[:first_12hr_idx - pd.Timedelta(hours=12)
                             ].drop(columns='time_diff').copy()
    df_12hr_segment = df.loc[first_12hr_idx:].drop(columns='time_diff').copy()
    df.drop(columns='time_diff', inplace=True)

    assert df_24hr_segment.shape[0] + df_12hr_segment.shape[0] == df.shape[0]

    if df_24hr_segment.empty:
        print("No 24-hour interval to upsample. Returning original data.")
        return df_12hr_segment.sort_index()

    # Calculate average proportions from the 12-hour segment
    df_12hr_segment['date_only'] = df_12hr_segment.index.normalize()
    daily_totals_12hr_segment = df_12hr_segment.groupby('date_only').sum()

    df_12hr_segment_with_daily_total = df_12hr_segment.merge(
        daily_totals_12hr_segment,
        left_on='date_only',
        right_index=True,
        suffixes=('', '_daily_total')
    )

    # Initialize a DataFrame to store the average proportions for 00:00, 12:00
    # The columns of this DataFrame will be your original numerical columns
    proportion_df = pd.DataFrame(
        index=['00:00', '12:00'], columns=df_24hr_segment.columns)

    for col in df_24hr_segment.columns:
        # Filter out rows where the daily total for this specific column is
        # zero to avoid division by zero
        valid_rows = df_12hr_segment_with_daily_total[col +
                                                      '_daily_total'] != 0

        if valid_rows.any():
            df_temp_col = df_12hr_segment_with_daily_total[valid_rows].copy()
            df_temp_col['proportion'] = df_temp_col[col] / \
                df_temp_col[col + '_daily_total']

            # Group by hour to get the average proportion for 00:00 and 12:00
            avg_proportions = df_temp_col.groupby(df_temp_col.index.hour)[
                'proportion'].mean()

            if 0 in avg_proportions.index:
                proportion_df.loc['00:00', col] = avg_proportions[0]
            if 12 in avg_proportions.index:
                proportion_df.loc['12:00', col] = avg_proportions[12]
        else:
            # If all daily totals for a column are zero, default to an even
            # spread (0.5 for each 12-hour period)
            print("Invalid row found")
            proportion_df.loc['00:00', col] = 0.5
            proportion_df.loc['12:00', col] = 0.5

    # Fill any remaining NaNs (e.g., if a column had no data for a specific
    # hour across the entire 12hr segment)
    proportion_df = proportion_df.fillna(0.5)

    # Re-normalize proportions to ensure they sum exactly to 1 for each column,
    # accounting for potential floating point errors
    proportion_df = proportion_df.div(proportion_df.sum(axis=0), axis=1)

    del df_12hr_segment['date_only']

    # Upsample the 24-hour segment using the calculated proportions
    upsampled_24hr_indices = []
    for dt in df_24hr_segment.index:
        upsampled_24hr_indices.append(dt)
        upsampled_24hr_indices.append(dt + pd.Timedelta(hours=12))

    upsampled_indices = pd.DatetimeIndex(
        sorted(list(set(upsampled_24hr_indices))))
    df_24hr_upsampled = pd.DataFrame(
        index=upsampled_indices, columns=df_24hr_segment.columns)

    for current_date_24hr in df_24hr_segment.index:
        original_24hr_values = df_24hr_segment.loc[current_date_24hr]
        ts_00 = current_date_24hr.replace(hour=0, minute=0, second=0)
        ts_12 = current_date_24hr.replace(hour=12, minute=0, second=0)

        for col in df_24hr_segment.columns:
            raw_value_00 = original_24hr_values[col] * \
                proportion_df.loc['00:00', col]

            if pd.isna(raw_value_00):
                value_00_final = value_12_final = np.nan
                print("Found NaN raw val")
            else:
                # Apply rounding: Round the first 12-hour segment
                value_00_rounded = int(round(raw_value_00))

                # Calculate the second 12-hour segment to ensure the sum
                # matches the original 24-hour total
                value_12_calculated = int(
                    original_24hr_values[col] - value_00_rounded)

                # Ensure values are non-negative (counts cannot be negative)
                value_00_final = max(0, value_00_rounded)
                value_12_final = max(0, value_12_calculated)

            if ts_00 in df_24hr_upsampled.index:
                df_24hr_upsampled.loc[ts_00, col] = value_00_final
            if ts_12 in df_24hr_upsampled.index:
                df_24hr_upsampled.loc[ts_12, col] = value_12_final

    # Combine and return the final DataFrame
    df_final_upsampled = pd.concat(
        [df_24hr_upsampled, df_12hr_segment]).sort_index()

    # Ensure final columns are numeric, handling concatenation
    for col in df_final_upsampled.columns:
        df_final_upsampled[col] = pd.to_numeric(
            df_final_upsampled[col], errors='coerce')

    return df_final_upsampled.astype(int)


def process_files(file_paths: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"\nWarning: File not found at '{file_path}'. Skipping.")
            continue

        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        print(f"\nProcessing '{file_name}'...")

        try:
            # Load and preprocess the data
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            if 'day' in df.columns:
                df.drop(columns='day', inplace=True)
            df.set_index('date', inplace=True)

            # Impute total columns where they are zero
            patient_cols = ['zone_a_mwr_patients',
                            'zone_a__patients', 'zone_b/c_patients']
            df.loc[df['total_number_of_patients'] == 0,
                   'total_number_of_patients'] = df[patient_cols].sum(axis=1)

            emews_cols = ['zone_a_mwr_sets_of_emews',
                          'zone_a__sets_of_emews', 'zone_b/c_sets_of_emews']
            df.loc[df['total_number_of_emews'] == 0,
                   'total_number_of_emews'] = df[emews_cols].sum(axis=1)

            # Perform the upsampling
            df_upsampled = upsample_dataframe(df)

            # Save the result
            df_upsampled.to_csv(output_path, index=True, index_label='date')
            print(f"Successfully processed and saved to '{output_path}'")

        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")


if __name__ == '__main__':
    INPUT_DIRECTORY = 'data/imputed'
    OUTPUT_DIRECTORY = 'data/upsampled'

    file_names = [
        'mean', 'median', 'mode', 'mice', 'mice_lr', 'mice_knn_5',
        'mice_knn_5_distance'
    ]

    files_to_process = [os.path.join(
        INPUT_DIRECTORY, f'{fname}_df.csv') for fname in file_names]

    # --- Run the processing ---
    if files_to_process:
        process_files(files_to_process, OUTPUT_DIRECTORY)
    else:
        print("The 'files_to_process' list is empty.")
