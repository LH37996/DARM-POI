import pandas as pd
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tqdm.contrib import tzip


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


def get_train_test():
    # Load the provided CSV file
    file_path = 'data/data_florida/aggregated_florida_visits.csv'
    florida_data = pd.read_csv(file_path)

    # Display the first few rows of the dataframe to understand its structure
    florida_data.head()

    # Extracting unique region ids and their corresponding latitudes and longitudes
    unique_regions = florida_data[['region_id', 'center_latitude', 'center_longitude']].drop_duplicates()
    region_ids = unique_regions['region_id'].values
    coordinates = unique_regions[['center_latitude', 'center_longitude']].values

    # Determining the number of clusters for KMeans (ideally 2 clusters for training and testing)
    n_clusters = 2

    # Applying KMeans clustering based on geographic coordinates
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coordinates)
    unique_regions['cluster'] = kmeans.labels_

    # Separating the regions into two groups based on the clustering
    train_regions = unique_regions[unique_regions['cluster'] == 0]['region_id'].tolist()
    test_regions = unique_regions[unique_regions['cluster'] == 1]['region_id'].tolist()

    # Saving the region IDs to JSON files
    train_regions_file = 'data/data_florida/train_regs.json'
    test_regions_file = 'data/data_florida/test_regs.json'

    with open(train_regions_file, 'w') as file:
        json.dump(train_regions, file)

    with open(test_regions_file, 'w') as file:
        json.dump(test_regions, file)

    # Plotting the regions
    plt.figure(figsize=(10, 6))

    # Plotting training regions
    train_coords = unique_regions[unique_regions['region_id'].isin(train_regions)][
        ['center_latitude', 'center_longitude']]
    plt.scatter(train_coords['center_longitude'], train_coords['center_latitude'], color='blue',
                label='Training Regions')

    # Plotting testing regions
    test_coords = unique_regions[unique_regions['region_id'].isin(test_regions)][
        ['center_latitude', 'center_longitude']]
    plt.scatter(test_coords['center_longitude'], test_coords['center_latitude'], color='red', label='Testing Regions')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographical Distribution of Regions in Florida (Training vs Testing)')
    plt.legend()
    plt.grid(True)
    plt.show()

    train_regions_file, test_regions_file, train_regions, test_regions


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


def get_poi_lat_long_list(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 按item_id排序
    df.sort_values('item_id', inplace=True)

    # 提取纬度和经度列
    poi_latitude_list = df['center_latitude'].tolist()
    poi_longitude_list = df['center_longitude'].tolist()

    # 提取item_id列
    poi_index_list = df['item_id'].tolist()

    # 将item_id保存到JSON文件
    json_file_path = 'data/data_florida/aggregated_poi_index_list.json'
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


def disaster_intensity_mapping(observation_intensity_list, observation_latitude_list, observation_longitude_list,
                               poi_latitude_list, poi_longitude_list
                               , s=1, k=1):
    assert len(observation_intensity_list) == len(observation_latitude_list) == len(observation_longitude_list)
    assert len(poi_latitude_list) == len(poi_longitude_list)
    """
    Calculate the intensity at each POI based on the observation data from weather stations.
    """
    num_pois = len(poi_latitude_list)
    num_stations = len(observation_intensity_list)

    # Create a matrix of distances between each POI and each observation station
    distance_matrix = np.zeros((num_pois, num_stations))
    for i, (poi_lat, poi_lon) in tqdm(enumerate(tzip(poi_latitude_list, poi_longitude_list))):
        for j, (obs_lat, obs_lon) in enumerate(zip(observation_latitude_list, observation_longitude_list)):
            distance_matrix[i, j] = haversine(poi_lon, poi_lat, obs_lon, obs_lat)

    # Inverse distance weighting
    weight_matrix = 1 / (distance_matrix + 1)

    # Convert observation_intensity_list to a column vector for element-wise multiplication
    observation_intensity_vector = np.array(observation_intensity_list).reshape(1, num_stations)

    # Element-wise product of weight_matrix and observation_intensity_vector, then sum across columns
    weighted_intensity = np.sum(weight_matrix * observation_intensity_vector, axis=1)

    # Apply the decay function
    poi_intensity_list = 1 + (k * weighted_intensity ** s)

    return poi_intensity_list


def add_intensity_to_csv(csv_file_path, intensity_list):
    df = pd.read_csv(csv_file_path)

    # 确保intensity_list的长度与CSV文件中的行数相同
    if len(intensity_list) != len(df):
        raise ValueError("Length of intensity_list does not match the number of rows in the CSV file.")

    # 将intensity_list添加为新列
    df['Intensity'] = intensity_list

    # 保存修改后的数据为新的CSV文件
    new_csv_file_path = 'data/aggregated_florida_visits_with_intensity.csv'
    df.to_csv(new_csv_file_path, index=False)

    return new_csv_file_path


def get_poi_intensity():
    observation_intensity_list, observation_latitude_list, observation_longitude_list = (
        process_weather_data_to_list("data/data_florida/daily_summaries_latest_filtered_wsf2")
    )
    poi_latitude_list, poi_longitude_list = get_poi_lat_long_list("data/data_florida/aggregated_florida_visits.csv")
    poi_intensity_list = disaster_intensity_mapping(observation_intensity_list, observation_latitude_list, observation_longitude_list,
                               poi_latitude_list, poi_longitude_list)
    add_intensity_to_csv("data/data_florida/aggregated_florida_visits.csv")
    return poi_intensity_list


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

    # get_train_test()

    get_poi_intensity()
