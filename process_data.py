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
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import multiprocessing


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


def aggregate_poi_visits(file_path, lat_delta, long_delta, bs):
    # 读取数据
    data = pd.read_csv(file_path)

    # 创建区域边界
    lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
    long_min, long_max = data['longitude'].min(), data['longitude'].max()

    # 计算区域划分
    lat_bins = np.arange(lat_min, lat_max, lat_delta)
    long_bins = np.arange(long_min, long_max, long_delta)

    # 为每个POI分配区域编号
    data['region_id'] = (
        np.digitize(data['latitude'], lat_bins) * 1000 +
        np.digitize(data['longitude'], long_bins)
    )

    # 计算每个区域编号对应区域中心的经纬度
    region_centers = data.groupby('region_id')[['latitude', 'longitude']].mean().reset_index()
    region_centers.rename(columns={'latitude': 'center_latitude', 'longitude': 'center_longitude'}, inplace=True)

    def get_train_test(unique_regions):
        coordinates = unique_regions[['center_latitude', 'center_longitude']].values

        # 仍然使用两个聚类
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

        # 如果训练集大小不是测试集的大约三倍，则进行调整
        while len(train_regions) < 3 * len(test_regions):
            # 从测试集中移动一些区域到训练集
            train_regions.append(test_regions.pop())

        # Saving the region IDs to JSON files
        train_regions_file = 'data/data_florida/train_regs_region.json'
        test_regions_file = 'data/data_florida/test_regs_region.json'

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
        plt.scatter(test_coords['center_longitude'], test_coords['center_latitude'], color='red',
                    label='Testing Regions')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geographical Distribution of Regions in Florida (Training vs Testing)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 在 split_and_balance_groups 函数之前获取 unique_regions
    unique_regions = region_centers[['region_id', 'center_latitude', 'center_longitude']].drop_duplicates()

    get_train_test(unique_regions)

    # 读取训练和测试区域的编号
    with open("data/data_florida/train_regs_region.json", 'r') as file:
        train_regs = json.load(file)
    with open("data/data_florida/test_regs_region.json", 'r') as file:
        test_regs = json.load(file)

    # 为每个数据点增加isTrain属性
    data['isTrain'] = data['region_id'].apply(lambda x: 1 if x in train_regs else 0)

    # 新逻辑：根据训练区域的数据分组
    def split_and_balance_groups(group):
        # 分离训练和测试数据
        train_group = group[group['isTrain'] == 1]
        test_group = group[group['isTrain'] == 0]

        # 分别对训练和测试数据进行分组
        for sub_group in [train_group, test_group]:
            total_size = len(sub_group)
            sizes = np.array([total_size // bs] * bs)
            sizes[:total_size % bs] += 1
            indices = np.repeat(range(bs), sizes)
            np.random.shuffle(indices)
            sub_group['bs'] = indices[:total_size]

        # 合并训练和测试数据回到原始组
        return pd.concat([train_group, test_group]).sample(frac=1)  # 随机打乱顺序

    data = data.groupby('region_id').apply(split_and_balance_groups).reset_index(drop=True)

    # 在每个组内按照主要类别聚合数据
    monthly_columns = [col for col in data.columns if col.startswith('20')]
    aggregated_data = data.groupby(['region_id', 'bs', 'top_category'])[monthly_columns].sum().reset_index()

    # 将区域中心的经纬度和isTrain属性合并进aggregated_data
    aggregated_data = pd.merge(aggregated_data, region_centers, on='region_id')
    aggregated_data = pd.merge(aggregated_data, data[['region_id', 'isTrain']].drop_duplicates(), on='region_id')

    # 重命名列和重新编号数据项
    aggregated_data.rename(columns={'latitude': 'center_latitude', 'longitude': 'center_longitude'}, inplace=True)
    aggregated_data['item_id'] = range(len(aggregated_data))

    # 保存聚合后的数据到新的CSV文件
    aggregated_csv_path = 'data/data_florida/aggregated_florida_visits.csv'
    aggregated_data.to_csv(aggregated_csv_path, index=False)

    def drop_data_to_ensure_same_count_of_train_test_ids():
        data = pd.read_csv('data/data_florida/aggregated_florida_visits.csv')
        # Step 1: Equalize the count of 'isTrain = 1' for each 'bs'
        # Filtering the data where isTrain = 1
        train_data = data[data['isTrain'] == 1]
        # Finding the minimum count of 'isTrain = 1' for any 'bs'
        min_train_count = train_data['bs'].value_counts().min()
        # Sampling the data to ensure equal count for each 'bs'
        balanced_train_data = train_data.groupby('bs').sample(n=min_train_count, random_state=1)
        # Step 2: Equalize the count of 'isTrain = 0' for each 'bs'
        # Filtering the data where isTrain = 0
        test_data = data[data['isTrain'] == 0]
        # Finding the minimum count of 'isTrain = 0' for any 'bs'
        min_test_count = test_data['bs'].value_counts().min()
        # Sampling the data to ensure equal count for each 'bs'
        balanced_test_data = test_data.groupby('bs').sample(n=min_test_count, random_state=1)
        # Step 3: Concatenate the balanced datasets and sort them
        # Concatenating the balanced datasets
        balanced_data = pd.concat([balanced_train_data, balanced_test_data])
        # Sorting the data first by 'bs' and then by 'isTrain'
        sorted_data = balanced_data.sort_values(by=['bs', 'isTrain'], ascending=[True, False])
        # Resetting the index for the sorted data
        sorted_data.reset_index(drop=True, inplace=True)
        # Step 4: Resetting the 'item_id' attribute
        # Resetting the 'item_id' starting from 0
        sorted_data['item_id'] = np.arange(len(sorted_data))
        output_file_path = 'data/data_florida/aggregated_florida_visits.csv'
        sorted_data.to_csv(output_file_path, index=False)

    drop_data_to_ensure_same_count_of_train_test_ids()

    # 记录相邻区域信息
    region_ids = aggregated_data['region_id'].unique()
    with open('data/data_florida/kg_region.txt', 'w') as f:
        for i, rid1 in enumerate(region_ids):
            for rid2 in region_ids[i + 1:]:
                # 判断区域是否相邻
                if abs(rid1 // 1000 - rid2 // 1000) <= 1 and abs(rid1 % 1000 - rid2 % 1000) <= 1:
                    f.write(f"{rid1}\tNearBy\t{rid2}\n")


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


def florida_visits_filter():
    # 洗florida的POI访问数据
    filtered_florida_visits = filter_non_zero_rows(2020, 1, 7)
    filtered_file_path = 'data/data_florida/Florida_visits_filtered.csv'
    # Save the filtered data to a new CSV file
    filtered_florida_visits.to_csv(filtered_file_path, index=False)


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


observation_intensity_list, observation_latitude_list, observation_longitude_list = (
    process_weather_data_to_list("data/data_florida/daily_summaries_latest_filtered_wsf2")
)

def calculate_poi_intensity(poi):
    poi_lat, poi_lon = poi
    poi_intensity = 0.0
    for (obs_lat, obs_lon) in zip(observation_latitude_list, observation_longitude_list):
        poi_intensity += 1 / (1 + haversine(poi_lon, poi_lat, obs_lon, obs_lat))
    # print(poi_intensity)
    return poi_intensity


def disaster_intensity_mapping(
                               poi_latitude_list, poi_longitude_list
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
    new_csv_file_path = 'data/data_florida/aggregated_florida_visits_with_intensity.csv'
    df.to_csv(new_csv_file_path, index=False)

    return new_csv_file_path


def get_poi_intensity():
    poi_latitude_list, poi_longitude_list = get_poi_lat_long_list("data/data_florida/aggregated_florida_visits.csv")
    poi_intensity_list = disaster_intensity_mapping(

        poi_latitude_list, poi_longitude_list
    )
    add_intensity_to_csv("data/data_florida/aggregated_florida_visits.csv", poi_intensity_list)
    return poi_intensity_list

def get_poi_feature_add_to_csv():
    file_path = 'data/data_florida/aggregated_florida_visits_with_intensity.csv'
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
    new_file_path = 'data/data_florida/aggregated_florida_visits_with_feature.csv'

    # Step 4: Save the updated data to a new CSV file
    data.to_csv(new_file_path, index=False)


def process_kg():
    kg_region_data = pd.read_csv('data/data_florida/kg_region.txt', sep='\t', header=None,
                                 names=['region_id_1', 'relation', 'region_id_2'])
    florida_visits_data = pd.read_csv('data/data_florida/aggregated_florida_visits.csv')

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


def replace_region_id_with_item_id(json_file_path):
    csv_file_path = "data/data_florida/aggregated_florida_visits.csv"
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


def generate_train_regs_list(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Filter data
    train_data = data[data['isTrain'] == 1]
    test_data = data[data['isTrain'] == 0]

    # Calculate n_train (number of 'isTrain = 1' entries for each 'bs')
    n_train_min = train_data['bs'].value_counts().min()
    n_train_max = train_data['bs'].value_counts().max()
    assert n_train_min == n_train_max
    n_train = n_train_min

    n_test_min = test_data['bs'].value_counts().min()
    n_test_max = test_data['bs'].value_counts().max()
    assert n_test_min == n_test_max
    n_test = n_test_min

    # Generate the list
    train_regs_list = list(range(n_train))
    test_regs_list = list(range(n_train, n_train + n_test))

    # Saving the list to a JSON file
    output_train_json_path = 'data/data_florida/train_regs.json'
    output_test_json_path = 'data/data_florida/test_regs.json'
    with open(output_train_json_path, 'w') as f:
        json.dump(train_regs_list, f)
    with open(output_test_json_path, 'w') as f:
        json.dump(test_regs_list, f)


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


# Start with: Florida_visits_2019_2020.csv, daily_summaries_latest_filtered_wsf2
def process_data(lat_delta, long_delta, bs):
    # Florida_visits_2019_2020.csv -> Florida_visits_filtered.csv
    florida_visits_filter()

    # Florida_visits_filtered.csv -> aggregated_florida_visits.csv
    aggregate_poi_visits("data/data_florida/Florida_visits_filtered.csv",
                         lat_delta, long_delta, bs)

    # aggregated_florida_visits.csv -> aggregated_florida_visits.csv
    convert_file_as_bs_order("data/data_florida/aggregated_florida_visits.csv")

    # aggregated_florida_visits.csv -> aggregated_florida_visits_with_intensity.csv
    get_poi_intensity()

    # aggregated_florida_visits_with_intensity.csv -> aggregated_florida_visits_with_feature.csv
    get_poi_feature_add_to_csv()

    process_kg()

    generate_train_regs_list("data/data_florida/aggregated_florida_visits.csv")


if __name__ == '__main__':
    process_data(2, 2, 100)

    # print(len(aggregate_and_plot_visits('data/data_florida/Florida_visits_filtered.csv',
    #                                     lat_delta=0.01, long_delta=0.01, plot=1)))

    # aggregate_poi_visits("data/data_florida/Florida_visits_filtered.csv",
    #                      lat_delta=3, long_delta=3, bs=100)

    # daily_summaries_latest_filter()

    # # 统计wsf2非空的文件数
    # files_with_wsf2, total_files = count_wsf2_in_files()
    # percentage = (files_with_wsf2 / total_files) * 100 if total_files > 0 else 0
    # print(f"WSF2 非空的文件数: {files_with_wsf2}/{total_files} ({percentage:.2f}%)")

    # count_wsf2_items()

    # filter_csv_files()

    # count_csv_data_items("data/data_florida/daily_summaries_latest_filtered_wsf2")

    # get_train_test()

    # get_poi_intensity()

    # get_poi_feature_add_to_csv()

    # process_kg()
    #
    # replace_region_id_with_item_id("data/data_florida/train_regs_region.json")
    # replace_region_id_with_item_id("data/data_florida/test_regs_region.json")
