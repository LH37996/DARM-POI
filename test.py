from load_data import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def load_flow_test_florida():
    data = pd.read_csv("data/data_florida/aggregated_florida_visits.csv")
    visit_data_columns = data.columns[2:26]  # 提取从2019年1月到2020年12月的访问数据列
    # category_columns = data.columns[2: 3]
    visit_data = data[visit_data_columns].values.tolist()  # 将访问数据转换为二维列表
    # category = data[category_columns].values.tolist()
    train_data = np.array(visit_data)
    M, m = np.max(train_data), np.min(train_data)
    train_data = (2 * train_data - m - M) / (M - m)  # 归一化到 [-1, 1]

    # # 应用对数变换
    # # 由于数据可能包含零或负值，我们在取对数之前加一个小常数
    # log_data = np.log(train_data + 1)
    # # 计算中位数和IQR
    # median = np.median(log_data)
    # Q1 = np.percentile(log_data, 25)
    # Q3 = np.percentile(log_data, 75)
    # IQR = Q3 - Q1
    # # 应用IQR标准化
    # normalized_data = (log_data - median) / IQR

    return train_data.tolist()
    # return visit_data, category


def load_flow_test_origin():
    with open('data/data_nyc/' + 'alldayflow.json', 'r') as f:
        date2flowmat = json.load(f)
    train_data = []
    for k, v in date2flowmat.items():
        if is_weekday(k):
            train_data.append(v)
    train_data = np.array(train_data)
    M, m = np.max(train_data), np.min(train_data)
    train_data = (2 * train_data - m - M) / (M - m)  # 归一化到 [-1, 1]
    print(train_data.tolist(), m, M)


def plot_visit_curve(r, data):
    """
    绘制第r个POI的访问量随时间变化的曲线
    参数:
    r -- POI的索引（从0开始）
    data -- 包含所有POI访问数据的二维列表
    """
    # 检查r是否在列表范围内
    if r < 0 or r >= len(data):
        print("错误：索引r超出范围。")
        return

    # 获取第r个POI的访问数据
    visit_data = data[r]

    # 时间标签（2019-01到2020-12）
    months = [f"{year}-{str(month).zfill(2)}" for year in range(2019, 2021) for month in range(1, 13)]

    # 绘制曲线
    plt.figure(figsize=(10, 5))
    plt.plot(months, visit_data, marker='o')
    plt.title(f"POI {r} Monthly Visitation Curve")
    plt.xlabel("Time(Month)")
    plt.ylabel("Visits Volume")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def calculate_lat_lon_range(csv_file_path):
    # 读取CSV文件
    data = pd.read_csv(csv_file_path)

    # 获取纬度和经度的最小值和最大值
    min_lat, max_lat = data['LATITUDE'].min(), data['LATITUDE'].max()
    min_lon, max_lon = data['LONGITUDE'].min(), data['LONGITUDE'].max()

    return min_lat, max_lat, min_lon, max_lon


# 示例：绘制POI的访问量变化曲线
def plot_poi_visitation_curve(r):
    (min_lat, max_lat, min_lon, max_lon) = calculate_lat_lon_range("data/data_florida/wsf2.csv")
    data = load_flow_test_florida()
    plot_visit_curve(r, data)
    # print(category[r])


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


def aggregate_poi_visits(file_path, lat_delta, long_delta):
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

    # 聚合数据
    monthly_columns = [col for col in data.columns if col.startswith('20')]
    aggregated_data = data.groupby(['region_id', 'top_category'])[monthly_columns].sum().reset_index()

    # 计算区域中心
    region_centers = data.groupby('region_id')['latitude', 'longitude'].mean().reset_index()
    aggregated_data = pd.merge(aggregated_data, region_centers, on='region_id')

    # 重命名列和重新编号数据项
    aggregated_data.rename(columns={'latitude': 'center_latitude', 'longitude': 'center_longitude'}, inplace=True)
    aggregated_data['item_id'] = range(1, len(aggregated_data) + 1)

    # 保存聚合后的数据到新的CSV文件
    aggregated_csv_path = 'data/data_florida/aggregated_florida_visits.csv'
    aggregated_data.to_csv(aggregated_csv_path, index=False)

    # 记录相邻区域信息
    region_ids = aggregated_data['region_id'].unique()
    with open('data/data_florida/kg.txt', 'w') as f:
        for i, rid1 in enumerate(region_ids):
            for rid2 in region_ids[i + 1:]:
                # 判断区域是否相邻
                if abs(rid1 // 1000 - rid2 // 1000) <= 1 and abs(rid1 % 1000 - rid2 % 1000) <= 1:
                    f.write(f"{rid1}\tNearBy\t{rid2}\n")

    return aggregated_csv_path, '/mnt/data/kg.txt'



if __name__ == "__main__":
    plot_poi_visitation_curve(603)

    # print(len(aggregate_and_plot_visits('data/data_florida/Florida_visits_filtered.csv',
    #                                     lat_delta=0.1, long_delta=0.1, plot=1)))

    # Example usage of the function
    # aggregate_poi_visits("data/data_florida/Florida_visits_filtered.csv",
    #                      lat_delta=0.01, long_delta=0.01)

    # print(load_flow_test_florida()[317687])
    # print(len([[1,2,3],[4,5,6]]))


