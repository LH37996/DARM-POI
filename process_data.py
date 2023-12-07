import pandas as pd
import os
import shutil


def daily_summaries_latest_filter():
    # 定义佛罗里达的经纬度范围
    florida_bounds = {
        "latitude_min": 24.545429,
        "latitude_max": 30.997623,
        "longitude_min": -87.518155,
        "longitude_max": -80.032537
    }

    # 文件夹路径
    input_folder = 'data/data_florida/daily-summaries-latest'
    output_folder = 'daily_summaries_latest_filtered'

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path, dtype={"DATE": "string", "LATITUDE": "string", "LONGITUDE": "string"})

            # 筛选数据
            filtered_df = df[
                (df['DATE'].str.startswith('2019') | df['DATE'].str.startswith('2020')) &
                (df['LATITUDE'].apply(lambda x: float(x)) >= florida_bounds['latitude_min']) &
                (df['LATITUDE'].apply(lambda x: float(x)) <= florida_bounds['latitude_max']) &
                (df['LONGITUDE'].apply(lambda x: float(x)) >= florida_bounds['longitude_min']) &
                (df['LONGITUDE'].apply(lambda x: float(x)) <= florida_bounds['longitude_max'])
            ]

            # 如果有符合条件的数据，将其保存到新文件夹
            if not filtered_df.empty:
                filtered_df.to_csv(os.path.join(output_folder, file), index=False)


def count_wsf2_in_files():
    # 替换为您的文件夹路径
    folder_path = 'data/data_florida/daily_summaries_latest_filtered'

    total_files = 0
    files_with_wsf2 = 0

    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # 检查 'WSF2' 列是否存在且不全为空
            if 'WSF2' in df.columns and not df['WSF2'].isna().all():
                files_with_wsf2 += 1

    return files_with_wsf2, total_files


def count_csv_data_items(folder_path):
    total_items = 0

    # Iterate through each file in the folder
    for file in os.listdir(folder_path):
        # Check if the file is a CSV
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                # Add the number of rows (data items) in this file to the total
                total_items += len(df)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

    print(total_items)


def count_wsf2_items():
    # 替换为您的文件夹路径
    folder_path = 'data/data_florida/daily_summaries_latest_filtered'
    total_items = 0

    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # 检查 'WSF2' 列是否存在且不全为空
            if 'WSF2' in df.columns and not df['WSF2'].isna().all():
                total_items += len(df)

    print(total_items)


def filter_csv_files():
    source_folder = "data/data_florida/daily_summaries_latest_filtered"
    destination_folder = "data/data_florida/daily_summaries_latest_filtered_wsf2"
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file in os.listdir(source_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(source_folder, file)
            try:
                df = pd.read_csv(file_path)

                # Check if there's at least one non-empty entry in 'WSF2' column
                if df['WSF2'].notna().any():
                    # Copy file to destination folder
                    shutil.copy(file_path, destination_folder)
            except Exception as e:
                print(f"Error processing file {file}: {e}")


def filter_non_zero_rows(year, start_month, end_month):
    """
    Filters rows where all the specified monthly columns are non-zero.

    :param df: DataFrame to be filtered.
    :param year: Year for which the months are considered.
    :param start_month: Starting month (inclusive).
    :param end_month: Ending month (inclusive).
    :return: Filtered DataFrame.
    """
    # Load the provided CSV file
    file_path = 'data/data_florida/Florida_visits_2019_2020.csv'
    df = pd.read_csv(file_path)

    # Create a list of column names for the specified months
    month_columns = [f"{year}-{str(month).zfill(2)}" for month in range(start_month, end_month + 1)]

    # Filter rows where all specified month columns are non-zero
    filtered_df = df[df[month_columns].all(axis=1)]

    return filtered_df


if __name__ == '__main__':
    # daily_summaries_latest_filter()

    # # 统计wsf2非空的文件数
    # files_with_wsf2, total_files = count_wsf2_in_files()
    # percentage = (files_with_wsf2 / total_files) * 100 if total_files > 0 else 0
    # print(f"WSF2 非空的文件数: {files_with_wsf2}/{total_files} ({percentage:.2f}%)")

    # count_wsf2_items()

    # filter_csv_files()

    # count_csv_data_items("data/data_florida/daily_summaries_latest_filtered_wsf2")

    # # 洗florida的POI访问数据
    # filtered_florida_visits = filter_non_zero_rows(2020, 1, 7)
    # filtered_file_path = 'data/data_florida/Florida_visits_filtered.csv'
    # # Save the filtered data to a new CSV file
    # filtered_florida_visits.to_csv(filtered_file_path, index=False)
