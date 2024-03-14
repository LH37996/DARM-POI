import pandas as pd
import os
import shutil
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
import csv

from dataset_config import dataset, running_baseline


def aggregate_and_plot_visits(florida_visits_path, lat_delta, long_delta, plot=1):
    # Load the POI data
    poi_data = pd.read_csv(florida_visits_path)

    # Determine the range of latitude and longitude
    lat_min, lat_max = poi_data['latitude'].min(), poi_data['latitude'].max()
    long_min, long_max = poi_data['longitude'].min(), poi_data['longitude'].max()

    # Calculate the number of bins for latitude and longitude
    lat_bins = np.arange(lat_min, lat_max, lat_delta)
    long_bins = np.arange(long_min, long_max, long_delta)

    # Create a DataFrame to store aggregated data
    aggregated_data = []

    # Iterate over each bin and aggregate visit data
    for lat in lat_bins:
        for long in long_bins:
            # Filter POIs within the current bin
            mask = (
                (poi_data['latitude'] >= lat) & (poi_data['latitude'] < lat + lat_delta) &
                (poi_data['longitude'] >= long) & (poi_data['longitude'] < long + long_delta)
            )
            filtered_data = poi_data[mask]

            # Sum visit data for each month
            if not filtered_data.empty:
                sum_data = filtered_data.iloc[:, 6:].sum()
                aggregated_data.append([lat, long] + sum_data.tolist())

    # Convert aggregated data to DataFrame
    column_names = ['latitude', 'longitude'] + [f'2019-{month:02d}' for month in range(1, 13)] + \
                   [f'2020-{month:02d}' for month in range(1, 13)]
    aggregated_df = pd.DataFrame(aggregated_data, columns=column_names)

    if plot == 1:
        # Plotting
        for index, row in aggregated_df.iterrows():
            plt.figure(figsize=(12, 6))
            row[2:].plot(kind='line')
            plt.title(f"POI Visits in Area Centered at (Lat: {row['latitude']}, Long: {row['longitude']}), "
                      f"$Lat Delta: {lat_delta}, Long Delta: {long_delta}")
            plt.xlabel('Month')
            plt.ylabel('Total Visits')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

    return aggregated_df


def get_train_test(file_path, data_folder_path):
    unique_regions = pd.read_csv(file_path)
    coordinates = unique_regions[['latitude', 'longitude']].values

    # 使用两个聚类
    n_clusters = 2

    # 应用 KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coordinates)
    unique_regions['cluster'] = kmeans.labels_

    # 根据聚类结果调整训练集和测试集的大小
    cluster_counts = unique_regions['cluster'].value_counts()
    larger_cluster = cluster_counts.idxmax()
    smaller_cluster = 1 - larger_cluster

    # 将较大的聚类分配给训练集，较小的聚类分配给测试集
    train_regions = unique_regions[unique_regions['cluster'] == larger_cluster]['region_id'].tolist()
    test_regions = unique_regions[unique_regions['cluster'] == smaller_cluster]['region_id'].tolist()

    # Saving the region IDs to JSON files
    train_regions_file = data_folder_path + '/train_regs_region.json'
    test_regions_file = data_folder_path + '/test_regs_region.json'

    with open(train_regions_file, 'w') as file:
        json.dump(train_regions, file)

    with open(test_regions_file, 'w') as file:
        json.dump(test_regions, file)

    # Plotting the regions
    plt.figure(figsize=(10, 6))

    # Plotting training regions
    train_coords = unique_regions[unique_regions['region_id'].isin(train_regions)][
        ['latitude', 'longitude']]
    plt.scatter(train_coords['longitude'], train_coords['latitude'], color='blue',
                label='Training Regions')

    # Plotting testing regions
    test_coords = unique_regions[unique_regions['region_id'].isin(test_regions)][
        ['latitude', 'longitude']]
    plt.scatter(test_coords['longitude'], test_coords['latitude'], color='red',
                label='Testing Regions')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographical Distribution of Regions (Training vs Testing)')
    plt.legend()
    plt.grid(True)
    plt.show()


def reorder_csv_and_add_isTrain(train_regions_path, test_regions_path, florida_visits_path, output_file_path):
    with open(train_regions_path, 'r') as file:
        train_regions = json.load(file)
    with open(test_regions_path, 'r') as file:
        test_regions = json.load(file)
    florida_visits_data = pd.read_csv(florida_visits_path)

    # Convert region_id lists to sets for faster lookup
    train_regions_set = set(train_regions)
    test_regions_set = set(test_regions)
    # Function to determine if a row belongs to the train or test set based on region_id
    def assign_is_train(row):
        if row['region_id'] in train_regions_set:
            return 1  # Indicates training data
        elif row['region_id'] in test_regions_set:
            return 0  # Indicates test data
        return None  # In case the region_id isn't found in either set
    # Apply the function to the dataframe
    florida_visits_data['isTrain'] = florida_visits_data.apply(assign_is_train, axis=1)
    # Split the data into train and test based on the new isTrain column
    train_data = florida_visits_data[florida_visits_data['isTrain'] == 1]
    test_data = florida_visits_data[florida_visits_data['isTrain'] == 0]
    # Concatenate the train and test data back together with train data first
    ordered_data = pd.concat([train_data, test_data])
    ordered_data.to_csv(output_file_path, index=False)


def add_region_id_and_save(input_file, output_file):
    """
    Add a 'region_id' column to the dataset, with a new ID for each unique latitude and longitude combination.
    Save the modified data to a new CSV file.

    Parameters:
    input_file (str): The path to the original CSV file.
    output_file (str): The path to save the modified CSV file with region IDs.
    """
    # Read the data
    df = pd.read_csv(input_file)

    # Create a dictionary to map unique latitude and longitude combinations to a new region_id
    unique_coords = {}
    region_id = 0

    # Iterate through each row and assign a region_id
    for idx, row in df.iterrows():
        coord = (row['latitude'], row['longitude'])
        if coord not in unique_coords:
            unique_coords[coord] = region_id
            region_id += 1
        df.at[idx, 'region_id'] = unique_coords[coord]

    # Save the modified dataframe to the new file
    df.to_csv(output_file, index=False)

    return f"Data with region_id added saved to {output_file}."


def add_item_id_and_save(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Add the 'item_id' column
    data.insert(0, 'item_id', range(len(data)))

    # Save the modified data back to the file
    data.to_csv(file_path, index=False)
    return data.head()  # Return the first few rows of the modified data for verification


# def aggregate_poi_visits(file_path, lat_delta, long_delta):
#     # 读取数据
#     data = pd.read_csv(file_path)
#
#     # 创建区域边界
#     lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
#     long_min, long_max = data['longitude'].min(), data['longitude'].max()
#
#     # 计算区域划分
#     lat_bins = np.arange(lat_min, lat_max, lat_delta)
#     long_bins = np.arange(long_min, long_max, long_delta)
#
#     # 为每个POI分配区域编号
#     data['region_id'] = (
#         np.digitize(data['latitude'], lat_bins) * 1000 +
#         np.digitize(data['longitude'], long_bins)
#     )
#
#     # 计算每个区域编号对应区域中心的经纬度
#     region_centers = data.groupby('region_id')[['latitude', 'longitude']].mean().reset_index()
#     region_centers.rename(columns={'latitude': 'center_latitude', 'longitude': 'center_longitude'}, inplace=True)
#
#     # 在 split_and_balance_groups 函数之前获取 unique_regions
#     unique_regions = region_centers[['region_id', 'center_latitude', 'center_longitude']].drop_duplicates()
#
#     get_train_test(unique_regions)
#
#     # 读取训练和测试区域的编号
#     with open("data/data_florida/train_regs_region.json", 'r') as file:
#         train_regs = json.load(file)
#     with open("data/data_florida/test_regs_region.json", 'r') as file:
#         test_regs = json.load(file)
#
#     # 为每个数据点增加isTrain属性
#     data['isTrain'] = data['region_id'].apply(lambda x: 1 if x in train_regs else 0)
#
#     # 在每个组内按照主要类别聚合数据
#     monthly_columns = [col for col in data.columns if col.startswith('20')]
#     aggregated_data = data.groupby(['region_id', 'top_category'])[monthly_columns].sum().reset_index()
#
#     # 将区域中心的经纬度和isTrain属性合并进aggregated_data
#     aggregated_data = pd.merge(aggregated_data, region_centers, on='region_id')
#     aggregated_data = pd.merge(aggregated_data, data[['region_id', 'isTrain']].drop_duplicates(), on='region_id')
#
#     # 保存聚合后的数据到新的CSV文件
#     aggregated_csv_path = 'data/data_florida/aggregated_florida_visits.csv'
#     aggregated_data.to_csv(aggregated_csv_path, index=False)
#
#     # aggregated_florida_visits.csv -> aggregated_florida_visits.csv
#     sort_data_by_train_test("data/data_florida/aggregated_florida_visits.csv",
#                             "data/data_florida/aggregated_florida_visits.csv")
#
#     aggregated_data = pd.read_csv("data/data_florida/aggregated_florida_visits.csv")
#
#     # 重命名列和重新编号数据项
#     aggregated_data.rename(columns={'latitude': 'center_latitude', 'longitude': 'center_longitude'}, inplace=True)
#     aggregated_data['item_id'] = range(len(aggregated_data))
#
#     # 保存聚合后的数据到新的CSV文件
#     aggregated_csv_path = 'data/data_florida/aggregated_florida_visits.csv'
#     aggregated_data.to_csv(aggregated_csv_path, index=False)
#
#     # 记录相邻区域信息
#     region_ids = aggregated_data['region_id'].unique()
#     with open('data/data_florida/kg_region.txt', 'w') as f:
#         for i, rid1 in enumerate(region_ids):
#             for rid2 in region_ids[i + 1:]:
#                 # 判断区域是否相邻
#                 if abs(rid1 // 1000 - rid2 // 1000) <= 1 and abs(rid1 % 1000 - rid2 % 1000) <= 1:
#                     f.write(f"{rid1}\tNearBy\t{rid2}\n")


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


# def filter_csv_files(data_folder_path):
#     source_folder = data_folder_path + "/daily_summaries_latest_filtered"
#     destination_folder = data_folder_path + "/daily_summaries_latest_filtered_wsf2"
#     # Create the destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     for file in os.listdir(source_folder):
#         if file.endswith('.csv'):
#             file_path = os.path.join(source_folder, file)
#             try:
#                 df = pd.read_csv(file_path)
#
#                 # Check if there's at least one non-empty entry in 'WSF2' column
#                 if df['WSF2'].notna().any():
#                     # Copy file to destination folder
#                     shutil.copy(file_path, destination_folder)
#             except Exception as e:
#                 print(f"Error processing file {file}: {e}")


def filter_and_copy_file(file, source_folder, destination_folder):
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

def filter_csv_files(data_folder_path):
    source_folder = data_folder_path + "/daily_summaries_latest_filtered"
    destination_folder = data_folder_path + "/daily_summaries_latest_filtered_wsf2"

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        tqdm(executor.map(filter_and_copy_file, files, [source_folder]*len(files), [destination_folder]*len(files)))


# # 新版本：实现了region对应
# def daily_summaries_latest_state_filter(data_folder_path, latitude_min, latitude_max, longitude_min, longitude_max):
#     # 定义FL或者SC的经纬度范围
#     florida_bounds = {
#         "latitude_min": latitude_min,
#         "latitude_max": latitude_max,
#         "longitude_min": longitude_min,
#         "longitude_max": longitude_max
#     }
#
#     # 文件夹路径
#     input_folder = data_folder_path + '/daily-summaries-latest'
#     output_folder = data_folder_path + 'daily_summaries_latest_filtered'
#
#     # 如果输出文件夹不存在，则创建它
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历文件夹中的所有文件
#     for file in os.listdir(input_folder):
#         if file.endswith('.csv'):
#             file_path = os.path.join(input_folder, file)
#             df = pd.read_csv(file_path, dtype={"DATE": "string", "LATITUDE": "string", "LONGITUDE": "string"})
#
#             # 筛选数据
#             filtered_df = df[
#                 (df['DATE'].str.startswith('2019') | df['DATE'].str.startswith('2020')) &
#                 (df['LATITUDE'].apply(lambda x: float(x)) >= florida_bounds['latitude_min']) &
#                 (df['LATITUDE'].apply(lambda x: float(x)) <= florida_bounds['latitude_max']) &
#                 (df['LONGITUDE'].apply(lambda x: float(x)) >= florida_bounds['longitude_min']) &
#                 (df['LONGITUDE'].apply(lambda x: float(x)) <= florida_bounds['longitude_max'])
#                 ]
#
#             # 如果有符合条件的数据，将其保存到新文件夹
#             if not filtered_df.empty:
#                 filtered_df.to_csv(os.path.join(output_folder, file), index=False)


def filter_file(file, input_folder, output_folder, florida_bounds):
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

def daily_summaries_latest_state_filter(data_folder_path, latitude_min, latitude_max, longitude_min, longitude_max):
    florida_bounds = {
        "latitude_min": latitude_min,
        "latitude_max": latitude_max,
        "longitude_min": longitude_min,
        "longitude_max": longitude_max
    }

    input_folder = data_folder_path + '/daily-summaries-latest'
    output_folder = data_folder_path + '/daily_summaries_latest_filtered'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    with ThreadPoolExecutor() as executor:
        tqdm(executor.map(filter_file, files, [input_folder]*len(files), [output_folder]*len(files), [florida_bounds]*len(files)))


def daily_summaries_latest_filter(data_folder_path):
    daily_summaries_latest_state_filter(data_folder_path,
                                        32.0346,  # latitude_min
                                        35.21554,  # latitude_max
                                        -83.35391,  # longitude_min
                                        -78.54203   # longitude_max
                                        )
    filter_csv_files(data_folder_path)

def filter_non_zero_rows(dataset, file_path, year, start_month=1, end_month=12):
    # Load the provided CSV file
    df = pd.read_csv(file_path)

    # Create a list of column names for the specified months
    if dataset == "florida":
        columns_to_check = [f"{year}-{str(month).zfill(2)}" for month in range(start_month, end_month + 1)]
    elif dataset == "FL_weekly":
        columns_to_check = [col for col in df.columns if col.startswith(str(year))]
    elif dataset == "SC_weekly":
        columns_to_check = [col for col in df.columns if col.startswith(str(year))]

    # Filter rows where all specified month columns are non-zero
    filtered_df = df[(df[columns_to_check] != 0.0).all(axis=1)]

    return filtered_df


def remove_rows_with_too_large_data(dataset, file_path):
    """
    Remove rows with any value greater than 100000 in the monthly visit columns
    and save the modified data back to the same CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    """
    # Read the data
    df = pd.read_csv(file_path)

    # Define the criteria for rows to keep (those with all values <= 100000 in the monthly visit columns)
    if dataset == "florida":
        criteria = (df.loc[:, '2019-01':'2019-12'] <= 10000).all(axis=1)
    elif dataset == "FL_weekly":
        criteria = (df.loc[:, '2019-01-07':'2019-09-30'] <= 10000).all(axis=1)
    elif dataset == "SC_weekly":
        criteria = (df.loc[:, '2018-01-01':'2018-10-08'] <= 10000).all(axis=1)

    # Keep only the rows that meet the criteria
    modified_df = df[criteria]

    # Save the modified dataframe back to the same file
    modified_df.to_csv(file_path, index=False)

    return f"Updated data saved to {file_path}. Removed {df.shape[0] - modified_df.shape[0]} rows."


def initial_file_filter(origin_path, target_path, data_count):
    # 洗florida的POI访问数据
    filtered_florida_visits = filter_non_zero_rows(dataset, origin_path, 2019)
    filtered_file_path = target_path
    # Save the filtered data to a new CSV file
    filtered_florida_visits.to_csv(filtered_file_path, index=False)
    remove_rows_with_too_large_data(dataset, filtered_file_path)

    # 减少数据
    file_path = target_path
    florida_visits = pd.read_csv(file_path)
    # Calculating the mean visitation for each place in 2019
    columns_used = []
    if dataset == "florida":
        columns_used = [col for col in florida_visits if col.startswith('2019')]
    elif dataset == "FL_weekly":
        columns_used = [col for col in florida_visits if (col.startswith('2019-01')
                                                       or col.startswith('2019-02')
                                                       or col.startswith('2019-03')
                                                       or col.startswith('2019-04')
                                                       or col.startswith('2019-05')
                                                       or col.startswith('2019-06')
                                                       or col.startswith('2019-07')
                                                       or col.startswith('2019-08'))]
    elif dataset == "SC_weekly":
        columns_used =[col for col in florida_visits if (col.startswith('2018-01')
                                                      or col.startswith('2018-02')
                                                      or col.startswith('2018-03')
                                                      or col.startswith('2018-04')
                                                      or col.startswith('2018-05')
                                                      or col.startswith('2018-06')
                                                      or col.startswith('2018-07')
                                                      or col.startswith('2018-08'))]
    florida_visits['normal_mean'] = florida_visits[columns_used].mean(axis=1)
    # Sorting the dataframe by the mean visitation in 2019 and retaining the top 200000 entries
    sorted_florida_visits = florida_visits.sort_values(by='normal_mean').head(data_count)
    # Saving the sorted dataframe to a new CSV file
    sorted_file_path = target_path
    sorted_florida_visits.to_csv(sorted_file_path, index=False)


def process_weather_station_data(weather_station_folder, florida_range):
    """
    Process the weather station data files.

    Parameters:
    - weather_station_folder: Path to the folder containing weather station files.
    - florida_range: Dictionary containing the latitude and longitude range for Florida.

    Returns:
    - observation_intensity_list: List of observation intensities for stations within Florida.
    - observation_latitude_list: List of latitudes for these stations.
    - observation_longitude_list: List of longitudes for these stations.
    """
    observation_intensity_list = []
    observation_latitude_list = []
    observation_longitude_list = []

    # Function to normalize values
    def normalize(values):
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)

    # Iterate through each weather station file
    for file in os.listdir(weather_station_folder):
        file_path = os.path.join(weather_station_folder, file)
        df = pd.read_csv(file_path)

        # Check if the station is within Florida's lat-lon range
        station_lat = df['LATITUDE'].iloc[0]
        station_lon = df['LONGITUDE'].iloc[0]
        if (florida_range['min_latitude'] <= station_lat <= florida_range['max_latitude'] and
                florida_range['min_longitude'] <= station_lon <= florida_range['max_longitude']):

            # Filter data for the first week of September 2019
            df['DATE'] = pd.to_datetime(df['DATE'])
            first_week_sept = df[(df['DATE'] >= datetime(2019, 9, 1)) & (df['DATE'] <= datetime(2019, 9, 7))]

            # Check if 'PRCP' and 'WSF2' have no missing values
            if first_week_sept['PRCP'].notna().all() and first_week_sept['WSF2'].notna().all():
                # Calculate mean values for 'PRCP' and 'WSF2'
                prcp_mean = first_week_sept['PRCP'].mean()
                wsf2_mean = first_week_sept['WSF2'].mean()

                # Normalize and calculate observation_intensity
                # normalized_values = normalize(np.array([prcp_mean, wsf2_mean]))
                normalized_values = np.array([prcp_mean, wsf2_mean])
                observation_intensity = np.sum(normalized_values)

                # Append the values to the lists
                observation_intensity_list.append(observation_intensity)
                observation_latitude_list.append(station_lat)
                observation_longitude_list.append(station_lon)

    return observation_intensity_list, observation_latitude_list, observation_longitude_list


def process_weather_data_to_list(folder_path):
    # 初始化列表和映射表
    station_to_new_id = {}
    data_list = []  # 用于存储所有数据，包括新序号
    new_id = 0

    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # 检查是否包含所需的列
        if set(['STATION', 'LATITUDE', 'LONGITUDE', 'PRCP', 'WSF2']).issubset(df.columns):
            # 如果新的STATION编号出现，则分配新的ID
            station_id = df['STATION'].iloc[0]
            if station_id not in station_to_new_id:
                station_to_new_id[station_id] = new_id
                new_id += 1

            # 归一化PRCP和WSF2
            scaler = MinMaxScaler()
            df[['PRCP', 'WSF2']] = scaler.fit_transform(df[['PRCP', 'WSF2']])

            # 计算灾害强度值
            df['Intensity'] = df['PRCP'] + df['WSF2']

            # 添加数据到列表
            for _, row in df.iterrows():
                data_list.append({
                    'new_id': station_to_new_id[station_id],
                    'intensity': row['Intensity'],
                    'latitude': row['LATITUDE'],
                    'longitude': row['LONGITUDE']
                })

    # 根据新序号排序
    data_list.sort(key=lambda x: x['new_id'])

    # 分离数据到各自的列表
    observation_intensity_list = [d['intensity'] for d in data_list]
    observation_latitude_list = [d['latitude'] for d in data_list]
    observation_longitude_list = [d['longitude'] for d in data_list]

    # 保存映射表到文件
    with open('data/data_florida/weather/station_mapping.json', 'w') as f:
        json.dump(station_to_new_id, f)

    # 保存列表到JSON文件
    with open('data/data_florida/weather/observation_intensity_list.json', 'w') as f:
        json.dump(observation_intensity_list, f)

    with open('data/data_florida/weather/observation_latitude_list.json', 'w') as f:
        json.dump(observation_latitude_list, f)

    with open('data/data_florida/weather/observation_longitude_list.json', 'w') as f:
        json.dump(observation_longitude_list, f)

    return observation_intensity_list, observation_latitude_list, observation_longitude_list


def get_poi_lat_long_list(csv_file_path, data_folder_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 按item_id排序
    df.sort_values('item_id', inplace=True)

    # 提取纬度和经度列
    poi_latitude_list = df['latitude'].tolist()
    poi_longitude_list = df['longitude'].tolist()

    # 提取item_id列
    poi_index_list = df['item_id'].tolist()

    # 将item_id保存到JSON文件
    json_file_path = data_folder_path + '/poi_index_list.json'
    with open(json_file_path, 'w') as file:
        json.dump(poi_index_list, file)

    return poi_latitude_list, poi_longitude_list


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Load the aggregated_florida_visits.csv file to get the latitude and longitude range
florida_visits_file = ""
if dataset == "florida":
    florida_visits_file = 'data/data_florida/Florida_visits_2019_2020.csv'
elif dataset == "FL_weekly":
    florida_visits_file = 'data/data_FL_weekly/FL_weekly_visits_2019.csv'
elif dataset == "SC_weekly":
    florida_visits_file = 'data/data_SC_weekly/SC_weekly_visits_2018_2019.csv'
florida_visits_df = pd.read_csv(florida_visits_file)
# Calculating the latitude and longitude range for Florida
min_latitude = florida_visits_df['latitude'].min()
max_latitude = florida_visits_df['latitude'].max()
min_longitude = florida_visits_df['longitude'].min()
max_longitude = florida_visits_df['longitude'].max()
florida_lat_lon_range = {
    "min_latitude": min_latitude,
    "max_latitude": max_latitude,
    "min_longitude": min_longitude,
    "max_longitude": max_longitude
}

if dataset == "florida":
    observation_intensity_list, observation_latitude_list, observation_longitude_list = (
        # process_weather_data_to_list("data/data_florida/daily_summaries_latest_filtered_wsf2")
        process_weather_station_data("data/data_florida/daily_summaries_latest_filtered_wsf2", florida_lat_lon_range)
    )
elif dataset == "FL_weekly":
    observation_intensity_list, observation_latitude_list, observation_longitude_list = (
        # process_weather_data_to_list("data/data_florida/daily_summaries_latest_filtered_wsf2")
        process_weather_station_data("data/data_FL_weekly/daily_summaries_latest_filtered_wsf2", florida_lat_lon_range)
    )
elif dataset == "SC_weekly":
    observation_intensity_list, observation_latitude_list, observation_longitude_list = (
        # process_weather_data_to_list("data/data_florida/daily_summaries_latest_filtered_wsf2")
        process_weather_station_data("data/data_SC_weekly/daily_summaries_latest_filtered_wsf2", florida_lat_lon_range)
    )

def calculate_poi_intensity(poi):
    poi_lat, poi_lon = poi
    poi_intensity = 0.0
    for (obs_lat, obs_lon) in zip(observation_latitude_list, observation_longitude_list):
        poi_intensity += 1 / (1 + haversine(poi_lon, poi_lat, obs_lon, obs_lat))
    # print(poi_intensity)
    return poi_intensity

def get_distance_list(poi):
    poi_lat, poi_lon = poi
    distance_list = []
    for (obs_lat, obs_lon) in zip(observation_latitude_list, observation_longitude_list):
        distance_list.append(haversine(poi_lon, poi_lat, obs_lon, obs_lat))
    return distance_list

def disaster_intensity_mapping(poi_latitude_list, poi_longitude_list
                               , s=1, k=1):
    assert len(observation_intensity_list) == len(observation_latitude_list) == len(observation_longitude_list)
    assert len(poi_latitude_list) == len(poi_longitude_list)
    """
    Calculate the intensity at each POI based on the observation data from weather stations.
    """
    num_pois = len(poi_latitude_list)
    num_stations = len(observation_intensity_list)

    # --- OUT OF MEMORY ---
    # Create a matrix of distances between each POI and each observation station
    # distance_matrix = np.zeros((num_pois, num_stations))
    # for i, (poi_lat, poi_lon) in tqdm(enumerate(tzip(poi_latitude_list, poi_longitude_list))):
    #     for j, (obs_lat, obs_lon) in enumerate(zip(observation_latitude_list, observation_longitude_list)):
    #         distance_matrix[i, j] = haversine(poi_lon, poi_lat, obs_lon, obs_lat)
    #
    # # Inverse distance weighting
    # weight_matrix = 1 / (distance_matrix + 1)
    #
    # # Convert observation_intensity_list to a column vector for element-wise multiplication
    # observation_intensity_vector = np.array(observation_intensity_list).reshape(1, num_stations)
    #
    # # Element-wise product of weight_matrix and observation_intensity_vector, then sum across columns
    # weighted_intensity = np.sum(weight_matrix * observation_intensity_vector, axis=1)
    #
    # # Apply the decay function
    # poi_intensity_list = 1 + (k * weighted_intensity ** s)

    # 单进程方法
    # poi_intensity_list = []
    # for (poi_lat, poi_lon) in tqdm(tzip(poi_latitude_list, poi_longitude_list)):
    #     poi_intensity = 0.0
    #     for (obs_lat, obs_lon) in zip(observation_latitude_list, observation_longitude_list):
    #         poi_intensity += 1 / (1 + haversine(poi_lon, poi_lat, obs_lon, obs_lat))
    #     poi_intensity_list.append(poi_intensity)

    poi_list = list(zip(poi_latitude_list, poi_longitude_list))
    # 使用多进程池处理外循环
    with ProcessPoolExecutor() as executor:
        poi_intensity_list = list(tqdm(
            executor.map(calculate_poi_intensity, poi_list), total=len(poi_list)
        ))

    return poi_intensity_list


def add_intensity_to_csv(csv_file_path, intensity_list):
    df = pd.read_csv(csv_file_path)

    # 确保intensity_list的长度与CSV文件中的行数相同
    if len(intensity_list) != len(df):
        raise ValueError("Length of intensity_list does not match the number of rows in the CSV file.")

    # 将intensity_list添加为新列
    df['Intensity'] = intensity_list

    # 保存修改后的数据为新的CSV文件
    new_csv_file_path = csv_file_path.replace(".csv", "_with_intensity.csv")
    df.to_csv(new_csv_file_path, index=False)

    return new_csv_file_path


def get_poi_intensity(file_path, data_folder_path):
    poi_latitude_list, poi_longitude_list = get_poi_lat_long_list(file_path, data_folder_path)
    poi_intensity_list = disaster_intensity_mapping(
        poi_latitude_list, poi_longitude_list
    )
    add_intensity_to_csv(file_path, poi_intensity_list)
    return poi_intensity_list


def distance_mapping(poi_latitude_list, poi_longitude_list):
    assert len(observation_intensity_list) == len(observation_latitude_list) == len(observation_longitude_list)
    assert len(poi_latitude_list) == len(poi_longitude_list)

    poi_list = list(zip(poi_latitude_list, poi_longitude_list))
    # 使用多进程池处理外循环
    with ProcessPoolExecutor() as executor:
        distances_list = list(tqdm(
            executor.map(get_distance_list, poi_list), total=len(poi_list)
        ))

    return distances_list


def add_distance_to_csv(csv_file_path, distances_list):
    df = pd.read_csv(csv_file_path)

    # 确保intensity_list的长度与CSV文件中的行数相同
    if len(distances_list) != len(df):
        raise ValueError("Length of intensity_list does not match the number of rows in the CSV file.")

    # 将intensity_list添加为新列
    df['Distance'] = distances_list

    # 保存修改后的数据为新的CSV文件
    new_csv_file_path = csv_file_path.replace(".csv", "_with_distance.csv")
    df.to_csv(new_csv_file_path, index=False)

    return new_csv_file_path


def get_distance(file_path, data_folder_path):
    poi_latitude_list, poi_longitude_list = get_poi_lat_long_list(file_path, data_folder_path)
    distances_list = distance_mapping(
        poi_latitude_list, poi_longitude_list
    )
    add_distance_to_csv(file_path, distances_list)
    return distances_list


def get_poi_feature_add_to_csv(file_path):
    data = pd.read_csv(file_path)
    # Step 1: Calculate the average visits from Jan 2020 to Mar 2020
    average_visits = data[['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08']].mean(axis=1)
    # Step 2: Normalize the average visits and the Intensity
    normalized_visits = (average_visits - average_visits.min()) / (average_visits.max() - average_visits.min())
    normalized_intensity = (data['Intensity'] - data['Intensity'].min()) / (
                data['Intensity'].max() - data['Intensity'].min())
    # Step 3: Create the new 'feature' column
    data['feature'] = list(zip(normalized_visits, normalized_intensity))
    # Prepare the file path for saving the new CSV
    new_file_path = file_path.replace("with_intensity.csv", "with_feature.csv")
    # Step 4: Save the updated data to a new CSV file
    data.to_csv(new_file_path, index=False)


def get_poi_feature_add_to_csv_distance(file_path):
    data = pd.read_csv(file_path)
    # Step 1: Calculate the average visits from Jan 2020 to Mar 2020
    columns_used = []
    if dataset == "florida":
        columns_used = [col for col in data if (col.startswith('2019-01')
                                             or col.startswith('2019-02')
                                             or col.startswith('2019-03')
                                             or col.startswith('2019-04')
                                             or col.startswith('2019-05')
                                             or col.startswith('2019-06')
                                             or col.startswith('2019-07')
                                             or col.startswith('2019-08'))]
    elif dataset == "FL_weekly":
        columns_used = [col for col in data if (col.startswith('2019-01')
                                             or col.startswith('2019-02')
                                             or col.startswith('2019-03')
                                             or col.startswith('2019-04')
                                             or col.startswith('2019-05')
                                             or col.startswith('2019-06')
                                             or col.startswith('2019-07')
                                             or col.startswith('2019-08'))]
    elif dataset == "SC_weekly":
        columns_used = [col for col in data if (col.startswith('2018-01')
                                             or col.startswith('2018-02')
                                             or col.startswith('2018-03')
                                             or col.startswith('2018-04')
                                             or col.startswith('2018-05')
                                             or col.startswith('2018-06')
                                             or col.startswith('2018-07')
                                             or col.startswith('2018-08'))]

    average_visits = data[columns_used].mean(axis=1)

    # Step 2: Normalize the average visits and the Intensity
    normalized_visits = (average_visits - average_visits.min()) / (average_visits.max() - average_visits.min())
    def to_list(data_D):
        res = []
        for s in data_D:
            res.append([float(item) for item in s.strip("[]").split(", ")])
        return res
    data_D = to_list(data['Distance'])
    min_data_D = min(value for inner_list in data_D for value in inner_list)
    max_data_D = max(value for inner_list in data_D for value in inner_list)
    normalized_distance = []
    for distances in data_D:
        normalized_distances = []
        for distance in distances:
            normalized_distances.append((distance - min_data_D) / (max_data_D - min_data_D))
        normalized_distance.append(normalized_distances)
    # Step 3: Create the new 'feature' column
    print(len(normalized_visits), len(normalized_distance), len(observation_intensity_list))
    observation_intensity_list_list = []
    for i in range(len(normalized_distance)):
        observation_intensity_list_list.append(observation_intensity_list)
    data['feature'] = list(zip(normalized_visits, normalized_distance, observation_intensity_list_list))
    # Prepare the file path for saving the new CSV
    new_file_path = file_path.replace("with_distance.csv", "with_feature.csv")
    # Step 4: Save the updated data to a new CSV file
    data.to_csv(new_file_path, index=False)


def process_kg(csv_file_path, path_to_kg_region):
    kg_region_data = pd.read_csv(path_to_kg_region, sep='\t', header=None,
                                 names=['region_id_1', 'relation', 'region_id_2'])
    florida_visits_data = pd.read_csv(csv_file_path)

    # Selecting a representative item_id for each region_id
    representative_item_ids = florida_visits_data.groupby('region_id')['item_id'].first()

    # Replace region_id in kg_region_data with their representative item_id
    kg_region_data['region_id_1'] = kg_region_data['region_id_1'].map(representative_item_ids)
    kg_region_data['region_id_2'] = kg_region_data['region_id_2'].map(representative_item_ids)

    # Prepare SameRegion relations
    same_region_relations = []
    for region_id, group in florida_visits_data.groupby('region_id'):
        representative_item_id = representative_item_ids[region_id]
        for item_id in group['item_id']:
            if item_id != representative_item_id:
                same_region_relations.append([item_id, 'SameRegion', representative_item_id])

    # Convert SameRegion relations to DataFrame
    same_region_df = pd.DataFrame(same_region_relations, columns=['item_id_1', 'relation', 'item_id_2'])

    # Combine NearBy and SameRegion relations
    combined_relations = pd.concat(
        [kg_region_data, same_region_df.rename(columns={'item_id_1': 'region_id_1', 'item_id_2': 'region_id_2'})])

    combined_relations.to_csv('data/data_florida/kg.txt', sep='\t', header=False, index=False)


def replace_region_id_with_item_id(json_file_path, csv_file_path):
    # Step 1: Read the JSON file to get the list of region_ids
    with open(json_file_path, 'r') as file:
        region_ids = json.load(file)
    # Step 2: Read the CSV file and create a mapping from region_id to item_id
    florida_data = pd.read_csv(csv_file_path)
    region_to_item = florida_data.groupby('region_id')['item_id'].apply(list).to_dict()
    # Step 3: Replace each region_id with corresponding item_ids
    item_ids = [item_id for region_id in region_ids for item_id in region_to_item.get(region_id, [])]
    # Step 4: Save the new list of item_ids as a JSON file
    new_json_file_path = json_file_path.replace('regs_region.json', 'regs.json')
    with open(new_json_file_path, 'w') as file:
        json.dump(item_ids, file)
    return new_json_file_path


# def generate_train_regs_list(file_path):
#     # Load the dataset
#     data = pd.read_csv(file_path)
#
#     # Filter data
#     train_data = data[data['isTrain'] == 1]
#     test_data = data[data['isTrain'] == 0]
#
#     # Calculate n_train (number of 'isTrain = 1' entries for each 'bs')
#     n_train = train_data.value_counts().min()
#
#     n_test = test_data.value_counts().min()
#
#     # Generate the list
#     train_regs_list = list(range(n_train))
#     test_regs_list = list(range(n_train, n_train + n_test))
#
#     # Saving the list to a JSON file
#     output_train_json_path = 'data/data_florida/train_regs.json'
#     output_test_json_path = 'data/data_florida/test_regs.json'
#     with open(output_train_json_path, 'w') as f:
#         json.dump(train_regs_list, f)
#     with open(output_test_json_path, 'w') as f:
#         json.dump(test_regs_list, f)


def convert_file_as_bs_order(file_path):
    data = pd.read_csv(file_path)
    def reorder_and_reindex(dataframe, sort_column, reindex_column):
        """
        Reorder the dataframe based on the values in sort_column (ascending).
        When values in sort_column are equal, original order is preserved.
        Reindex the reindex_column starting from 0.

        :param dataframe: Pandas DataFrame to be processed.
        :param sort_column: Column name (str) to sort by.
        :param reindex_column: Column name (str) to reindex.
        :return: Processed DataFrame.
        """
        # Sort the dataframe by the specified column, keeping the original order when values are the same
        sorted_df = dataframe.sort_values(by=sort_column, kind='mergesort')

        # Reindex the specified column starting from 0
        sorted_df[reindex_column] = range(len(sorted_df))

        return sorted_df

    # Apply the function to the data
    sorted_data = reorder_and_reindex(data, 'bs', 'item_id')
    sorted_data.to_csv(file_path, index=False)


def sort_data_by_train_test(file_path, output_file_path):
    import pandas as pd

    # 读取数据
    data = pd.read_csv(file_path)

    # 首先按照 region_id 和 isTrain 排序
    sorted_data = data.sort_values(by=['isTrain', 'region_id'], ascending=[False, True])

    # 保存排序后的数据到新文件
    sorted_data.to_csv(output_file_path, index=False)

    return output_file_path


def sort_train_test_regs(data_folder_path):
    train_file_path = data_folder_path + '/train_regs.json'
    test_file_path = data_folder_path + '/test_regs.json'

    # Load the first file (train_regs.json)
    with open(train_file_path, 'r') as file:
        train_data = json.load(file)

    # Load the second file (test_regs.json)
    with open(test_file_path, 'r') as file:
        test_data = json.load(file)

    # Sorting the data in ascending order
    train_data_sorted = sorted(train_data)
    test_data_sorted = sorted(test_data)

    # Writing the sorted data back to the original files

    with open(train_file_path, 'w') as file:
        json.dump(train_data_sorted, file)

    with open(test_file_path, 'w') as file:
        json.dump(test_data_sorted, file)


def get_kg(file_path, data_folder_path):
    florida_poi_data = pd.read_csv(file_path)
    # Extract relevant columns for Voronoi tessellation
    coordinates = florida_poi_data[['latitude', 'longitude']].values

    # Create Voronoi Tessellation
    vor = Voronoi(coordinates)

    # Function to find adjacent regions
    def find_adjacent_regions(vor):
        adjacency_list = {}
        for point_idx, ridge_points in enumerate(vor.ridge_points):
            region1, region2 = ridge_points
            if region1 not in adjacency_list:
                adjacency_list[region1] = set()
            if region2 not in adjacency_list:
                adjacency_list[region2] = set()
            adjacency_list[region1].add(region2)
            adjacency_list[region2].add(region1)
        return adjacency_list

    # Find adjacent regions
    adjacency_list = find_adjacent_regions(vor)

    # Convert adjacency_list to a format suitable for kg.txt
    kg_data = []
    for region, neighbors in tqdm(adjacency_list.items()):
        for neighbor in neighbors:
            kg_data.append(f"{region}\tNearBy\t{neighbor}")

    # Define the path for the kg.txt file
    kg_file_path = data_folder_path + '/kg.txt'

    # Write the adjacency information to the kg.txt file
    with open(kg_file_path, 'w') as file:
        for line in kg_data:
            file.write(line + '\n')


def remove_duplicate_kg_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    unique_pairs = set()
    processed_lines = []

    for line in lines:
        item_id_1, _, item_id_2 = line.strip().split('\t')
        pair = tuple(sorted([item_id_1, item_id_2]))

        if pair not in unique_pairs:
            unique_pairs.add(pair)
            processed_lines.append(line)

    # 将处理后的数据保存回文件
    with open(file_path, 'w') as file:
        file.writelines(processed_lines)


def sparsify_graph(file_path, fraction=0.01):
    """
    Function to sparsify a graph represented in a file.
    It retains only a fraction of the edges.

    :param file_path: Path to the file containing the graph.
    :param fraction: Fraction of edges to retain (default is 0.1).
    """
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Randomly select a fraction of the lines (edges)
    selected_lines = random.sample(lines, int(len(lines) * fraction))

    # Write the selected lines back to the file
    with open(file_path, 'w') as file:
        for line in selected_lines:
            file.write(line)


def get_ratio(file_path, output_path):
    # Step 1: Read the data
    data = pd.read_csv(file_path)

    # Step 2: Calculate the mean of POI visits
    columns_used = []
    if dataset == "florida":
        columns_used = [col for col in data if (col.startswith('2019-01')
                                             or col.startswith('2019-02')
                                             or col.startswith('2019-03')
                                             or col.startswith('2019-04')
                                             or col.startswith('2019-05')
                                             or col.startswith('2019-06')
                                             or col.startswith('2019-07')
                                             or col.startswith('2019-08'))]
    elif dataset == "FL_weekly":
        columns_used = [col for col in data if (col.startswith('2019-01')
                                             or col.startswith('2019-02')
                                             or col.startswith('2019-03')
                                             or col.startswith('2019-04')
                                             or col.startswith('2019-05')
                                             or col.startswith('2019-06')
                                             or col.startswith('2019-07')
                                             or col.startswith('2019-08'))]
    elif dataset == "SC_weekly":
        columns_used = [col for col in data if (col.startswith('2018-01')
                                             or col.startswith('2018-02')
                                             or col.startswith('2018-03')
                                             or col.startswith('2018-04')
                                             or col.startswith('2018-05')
                                             or col.startswith('2018-06')
                                             or col.startswith('2018-07')
                                             or col.startswith('2018-08'))]

    data['mean_Jan_to_Aug'] = data[columns_used].mean(axis=1)

    # Step 3: Update the values for September to December 2019
    if dataset == "florida":
        for month in range(9, 13):
            month_col = f"2019-{str(month).zfill(2)}"
            last_month_col = f"2019-{str(month - 1).zfill(2)}"
            data[month_col] = (data[month_col] - data[last_month_col]) / data['mean_Jan_to_Aug']
    elif dataset == "FL_weekly":
        for month in range(5):
            week_col = "2019-09-02"
            index = data.columns.get_loc(week_col)
            last_week_col = data.columns[index - 1]
            data[week_col] = (data[week_col] - data[last_week_col]) / data['mean_Jan_to_Aug']
    elif dataset == "SC_weekly":
        for month in range(4):
            week_col = "2018-09-17"
            index = data.columns.get_loc(week_col)
            last_week_col = data.columns[index - 1]
            data[week_col] = (data[week_col] - data[last_week_col]) / data['mean_Jan_to_Aug']

    # Step 4: Save the modified data to a new file
    data.to_csv(output_path, index=False)
    return "Processing completed. Data saved to: " + output_path


def remove_duplicates_in_train_test_and_save(file_path):
    """
    Function to remove duplicates from a list stored in a JSON file and save the results back to the same file.
    Parameters:
    file_path (str): Path to the JSON file containing the list.
    """
    # Load the list from the file
    with open(file_path, 'r') as file:
        data_list = json.load(file)

    # Remove duplicates
    data_list = list(set(data_list))

    # Save the list back to the file
    with open(file_path, 'w') as file:
        json.dump(data_list, file, indent=4)


def update_weekly_visits(visits_csv_path, weekly_data_dir):
    # Load the CSV file into a DataFrame
    florida_visits_df = pd.read_csv(visits_csv_path)

    # Initialize a new column 'Weekly' for storing weekly visits data
    florida_visits_df['Weekly'] = None
    for index in range(len(florida_visits_df)):  # Initialize empty lists for the specified number of records
        florida_visits_df.at[index, 'Weekly'] = []

    # Iterate over each file only once and update the weekly data for the specified placekeys if present
    extracted_subdir_files = os.listdir(weekly_data_dir)
    for file_name in extracted_subdir_files:
        if file_name.endswith('.csv'):  # Ensure to process only CSV files
            file_path = os.path.join(weekly_data_dir, file_name)
            with open(file_path, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                # Create a map of placekey to its index for quick lookup
                placekey_to_index = {florida_visits_df.at[i, 'placekey']: i for i in range(florida_visits_df)}
                for row in csv_reader:
                    placekey = row.get('placekey')
                    if placekey in placekey_to_index:
                        # Update weekly visits if 'normalized_visits_by_state_scaling' is not empty
                        normalized_visits = row.get('normalized_visits_by_state_scaling')
                        if normalized_visits:
                            index = placekey_to_index[placekey]
                            florida_visits_df.at[index, 'Weekly'].append(float(normalized_visits))

    # Save the updated DataFrame to a new CSV file
    updated_csv_path = visits_csv_path.replace('.csv', '_updated.csv')
    florida_visits_df.to_csv(updated_csv_path, index=False)

    return updated_csv_path


# Start with: Florida_visits_2019_2020.csv, daily_summaries_latest_filtered_wsf2
def process_data():
    if dataset == "florida":
        initial_file_filter('data/data_florida/Florida_visits_2019_2020.csv',
                            "data/data_florida/Florida_visits_filtered.csv",
                            30000)

        add_region_id_and_save("data/data_florida/Florida_visits_filtered.csv",
                               "data/data_florida/Florida_visits_filtered_with_region_id.csv")

        get_train_test("data/data_florida/Florida_visits_filtered_with_region_id.csv",
                       "data/data_florida")

        reorder_csv_and_add_isTrain("data/data_florida/train_regs_region.json",
                                    "data/data_florida/test_regs_region.json",
                                    "data/data_florida/Florida_visits_filtered_with_region_id.csv",
                                    "data/data_florida/Florida_visits_reordered_with_isTrain.csv")

        add_item_id_and_save("data/data_florida/Florida_visits_reordered_with_isTrain.csv")

        # # Florida_visits_filtered.csv -> aggregated_florida_visits.csv
        # aggregate_poi_visits("data/data_florida/Florida_visits_filtered.csv",
        #                      lat_delta, long_delta)

        # aggregated_florida_visits.csv -> aggregated_florida_visits_with_intensity.csv
        # get_poi_intensity("data/data_florida/Florida_visits_reordered_with_isTrain.csv",
        #                   "data/data_florida")
        # Florida_visits_reordered_with_isTrain.csv.csv -> Florida_visits_reordered_with_distance.csv
        get_distance("data/data_florida/Florida_visits_reordered_with_isTrain.csv",
                     "data/data_florida")


        # GHANGED AND CHANGE_THE_FUNCTION! aggregated_florida_visits_with_intensity.csv -> aggregated_florida_visits_with_feature.csv
        get_poi_feature_add_to_csv_distance("data/data_florida/Florida_visits_reordered_with_isTrain_with_distance.csv")

        # process_kg()
        get_kg("data/data_florida/Florida_visits_reordered_with_isTrain_with_intensity.csv",
               'data/data_florida')

        remove_duplicate_kg_data("data/data_florida/kg.txt")

        # sparsify_graph("data/data_florida/kg.txt")

        get_ratio("data/data_florida/Florida_visits_reordered_with_isTrain_with_feature.csv",
                  "data/data_florida/Florida_visits_reordered_with_isTrain_with_feature.csv")

        replace_region_id_with_item_id("data/data_florida/train_regs_region.json",
                                       "data/data_florida/Florida_visits_reordered_with_isTrain_with_feature.csv")
        replace_region_id_with_item_id("data/data_florida/test_regs_region.json",
                                       "data/data_florida/Florida_visits_reordered_with_isTrain_with_feature.csv")

        remove_duplicates_in_train_test_and_save("data/data_florida/train_regs.json")
        remove_duplicates_in_train_test_and_save("data/data_florida/test_regs.json")
        sort_train_test_regs("data/data_florida")

    elif dataset == "FL_weekly":
        initial_file_filter('data/data_FL_weekly/FL_weekly_visits_2019.csv',
                            "data/data_FL_weekly/FL_weekly_visits_filtered.csv",
                            30000)
        add_region_id_and_save("data/data_FL_weekly/FL_weekly_visits_filtered.csv",
                               "data/data_FL_weekly/FL_weekly_visits_filtered_with_region_id.csv")
        get_train_test("data/data_FL_weekly/FL_weekly_visits_filtered_with_region_id.csv",
                       "data/data_FL_weekly")
        reorder_csv_and_add_isTrain("data/data_FL_weekly/train_regs_region.json",
                                    "data/data_FL_weekly/test_regs_region.json",
                                    "data/data_FL_weekly/FL_weekly_visits_filtered_with_region_id.csv",
                                    "data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain.csv")
        add_item_id_and_save("data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain.csv")
        get_distance("data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain.csv",
                     "data/data_FL_weekly")
        get_poi_feature_add_to_csv_distance("data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain_with_distance.csv")
        get_kg("data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain_with_feature.csv",
               'data/data_FL_weekly')
        remove_duplicate_kg_data("data/data_FL_weekly/kg.txt")
        if not running_baseline:
            get_ratio("data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain_with_feature.csv",
                  "data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain_with_feature.csv")
        replace_region_id_with_item_id("data/data_FL_weekly/train_regs_region.json",
                                       "data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain_with_feature.csv")
        replace_region_id_with_item_id("data/data_FL_weekly/test_regs_region.json",
                                       "data/data_FL_weekly/FL_weekly_visits_reordered_with_isTrain_with_feature.csv")
        remove_duplicates_in_train_test_and_save("data/data_FL_weekly/train_regs.json")
        remove_duplicates_in_train_test_and_save("data/data_FL_weekly/test_regs.json")
        sort_train_test_regs("data/data_FL_weekly")

    elif dataset == "SC_weekly":
        initial_file_filter('data/data_SC_weekly/SC_weekly_visits_2018_2019.csv',
                            "data/data_SC_weekly/SC_weekly_visits_filtered.csv",
                            30000)
        add_region_id_and_save("data/data_SC_weekly/SC_weekly_visits_filtered.csv",
                               "data/data_SC_weekly/SC_weekly_visits_filtered_with_region_id.csv")
        get_train_test("data/data_SC_weekly/SC_weekly_visits_filtered_with_region_id.csv",
                       "data/data_SC_weekly")
        reorder_csv_and_add_isTrain("data/data_SC_weekly/train_regs_region.json",
                                    "data/data_SC_weekly/test_regs_region.json",
                                    "data/data_SC_weekly/SC_weekly_visits_filtered_with_region_id.csv",
                                    "data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain.csv")
        add_item_id_and_save("data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain.csv")
        get_distance("data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain.csv",
                     "data/data_SC_weekly")
        get_poi_feature_add_to_csv_distance(
            "data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain_with_distance.csv")
        get_kg("data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain_with_feature.csv",
               'data/data_SC_weekly')
        remove_duplicate_kg_data("data/data_SC_weekly/kg.txt")
        if not running_baseline:
            get_ratio("data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain_with_feature.csv",
                      "data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain_with_feature.csv")
        replace_region_id_with_item_id("data/data_SC_weekly/train_regs_region.json",
                                       "data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain_with_feature.csv")
        replace_region_id_with_item_id("data/data_SC_weekly/test_regs_region.json",
                                       "data/data_SC_weekly/SC_weekly_visits_reordered_with_isTrain_with_feature.csv")
        remove_duplicates_in_train_test_and_save("data/data_SC_weekly/train_regs.json")
        remove_duplicates_in_train_test_and_save("data/data_SC_weekly/test_regs.json")
        sort_train_test_regs("data/data_SC_weekly")


if __name__ == '__main__':
    process_data()