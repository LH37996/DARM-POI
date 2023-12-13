import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_weather_data(station_id):
    """
    Plots the PRCP (Precipitation) and WSF2 (Maximum wind speed) over time for a given weather station.

    Parameters:
    station_id (str): The identifier of the weather station.
    """
    # File path for the uploaded zip file
    zip_file_path = '/mnt/data/daily_summaries_latest_filtered_wsf2.zip'

    # Directory to extract the files
    extract_dir = '/mnt/data/daily_summaries/'
    extracted_files = os.listdir(extract_dir)
    nested_dir = os.path.join(extract_dir, extracted_files[0])

    # Constructing the path for the CSV file
    csv_file_path = os.path.join(nested_dir, f"{station_id}.csv")

    # Reading the CSV file
    try:
        data = pd.read_csv(csv_file_path, usecols=['DATE', 'PRCP', 'WSF2'])
    except FileNotFoundError:
        print(f"No data available for station ID: {station_id}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Convert DATE column to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Plotting
    fig, ax1 = plt.subplots()

    # Plot PRCP
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PRCP (mm)', color=color)
    ax1.plot(data['DATE'], data['PRCP'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('WSF2 (m/s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(data['DATE'], data['WSF2'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # To ensure the right y-label is not clipped
    plt.title(f"Weather Data for Station {station_id}")
    plt.show()


if __name__ == '__main__':
    plot_weather_data('USW00003818')