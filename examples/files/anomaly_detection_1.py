
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Assuming the data is in a CSV file named 'temperature_data.csv'
try:
    df = pd.read_csv('temperature_data.csv')
except FileNotFoundError:
    print("Error: temperature_data.csv not found.  Please ensure the file is in the correct directory.")
    exit()

# --- Data Cleaning and Preprocessing ---

# Convert year, month, day to datetime objects (Handle missing days/months if needed)
df['Date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1), errors='coerce')

# Fill missing temperature values (Simple imputation - can be improved)
df['AverageTemperatureFahr'].fillna(df['AverageTemperatureFahr'].mean(), inplace=True)
df['AverageTemperatureUncertaintyFahr'].fillna(df['AverageTemperatureUncertaintyFahr'].mean(), inplace=True) #Or median

# --- Anomaly Detection Functions ---

def detect_missing_data_anomalies(df, column, threshold=0.5):
    """
    Identifies years with a high percentage of missing data in a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to check for missing values.
        threshold (float): The threshold for the percentage of missing data (e.g., 0.5 for 50%).

    Returns:
        pd.DataFrame: A DataFrame containing years with a missing data percentage above the threshold.
    """
    missing_data_by_year = df.groupby('year')[column].apply(lambda x: x.isnull().sum() / len(x))
    anomalous_years = missing_data_by_year[missing_data_by_year > threshold].index.tolist()
    return anomalous_years

def detect_high_uncertainty_regions(df, uncertainty_column='AverageTemperatureUncertaintyFahr', percentile=95):
    """
    Identifies cities/countries with consistently high temperature uncertainty.

    Args:
        df (pd.DataFrame): The input DataFrame.
        uncertainty_column (str): The name of the column containing uncertainty values.
        percentile (int): The percentile to use as a threshold for high uncertainty.

    Returns:
        pd.DataFrame: A DataFrame of cities/countries with uncertainty above the specified percentile.
    """
    uncertainty_threshold = df[uncertainty_column].quantile(percentile / 100)
    high_uncertainty_locations = df.groupby(['City', 'Country'])[uncertainty_column].mean()
    anomalous_locations = high_uncertainty_locations[high_uncertainty_locations > uncertainty_threshold].index.tolist()
    return anomalous_locations

def detect_temperature_spikes(df, city, temperature_column='AverageTemperatureFahr', window=30, std_dev_threshold=3):
    """
    Detects temperature spikes/dips in a time series for a specific city using a moving average.

    Args:
        df (pd.DataFrame): The input DataFrame.
        city (str): The city to analyze.
        temperature_column (str): The name of the column containing temperature values.
        window (int): The window size for the moving average.
        std_dev_threshold (float): The number of standard deviations from the moving average to consider a spike/dip.

    Returns:
        pd.DataFrame: A DataFrame containing dates of detected temperature spikes/dips.
    """
    city_data = df[df['City'] == city].sort_values('Date')
    city_data['MovingAverage'] = city_data[temperature_column].rolling(window=window, center=True).mean() #Centered moving average
    city_data['StdDev'] = city_data[temperature_column].rolling(window=window, center=True).std()
    city_data['UpperThreshold'] = city_data['MovingAverage'] + std_dev_threshold * city_data['StdDev']
    city_data['LowerThreshold'] = city_data['MovingAverage'] - std_dev_threshold * city_data['StdDev']
    city_data['Spike'] = (city_data[temperature_column] > city_data['UpperThreshold']) | (city_data[temperature_column] < city_data['LowerThreshold'])
    spikes = city_data[city_data['Spike']][['Date', temperature_column, 'MovingAverage', 'StdDev']]
    return spikes

def detect_unusual_seasonal_patterns(df, city, temperature_column='AverageTemperatureFahr', comparison_year=2000):
    """
    Detects unusual seasonal patterns by comparing a year's temperature profile to a baseline year.

    Args:
        df (pd.DataFrame): The input DataFrame.
        city (str): The city to analyze.
        temperature_column (str): The name of the temperature column.
        comparison_year (int): The year to use as a baseline for seasonal comparison.

    Returns:
        pd.DataFrame: A DataFrame containing months with significantly different temperatures compared to the baseline year.
    """
    city_data = df[df['City'] == city]
    baseline_data = city_data[city_data['year'] == comparison_year].groupby('month')[temperature_column].mean()
    current_data = city_data.groupby('month')[temperature_column].mean()

    # Calculate the difference between each month's average temperature and the baseline
    temperature_diff = current_data - baseline_data

    # Identify months where the temperature difference is significant (e.g., using a threshold)
    threshold = temperature_diff.std() * 2  # Example threshold: 2 standard deviations

    # Filter the temperature differences to find the months that exceed the threshold
    significant_diff_months = temperature_diff[abs(temperature_diff) > threshold]
    
    return significant_diff_months

def detect_correlation_anomalies(df, temperature_column='AverageTemperatureFahr'):
    """
    Detects correlation-based anomalies by comparing temperature to latitude.

    Args:
        df (pd.DataFrame): The input DataFrame.
        temperature_column (str): The name of the temperature column.

    Returns:
        pd.DataFrame: A DataFrame containing data points with temperatures significantly deviating from the expected latitude-temperature relationship.
    """
    # Convert latitude to numeric and handle 'N' and 'S'
    df['LatitudeNumeric'] = df['Latitude'].str.replace('N', '').str.replace('S', '-').astype(float)

    # Linear regression model
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['LatitudeNumeric'], df[temperature_column])

    # Calculate expected temperature based on the linear model
    df['ExpectedTemperature'] = intercept + slope * df['LatitudeNumeric']

    # Calculate residuals (difference between actual and expected)
    df['Residuals'] = df[temperature_column] - df['ExpectedTemperature']

    # Define anomaly threshold based on standard deviation of residuals
    threshold = df['Residuals'].std() * 2  # Example: 2 standard deviations

    # Identify anomalies
    anomalies = df[abs(df['Residuals']) > threshold]
    return anomalies[['Date', 'City', 'Country', 'Latitude', temperature_column, 'ExpectedTemperature', 'Residuals']]


# --- Main Execution ---

# 1. Missing Data Anomalies
missing_data_anomalies = detect_missing_data_anomalies(df, 'AverageTemperatureFahr')
print("\nYears with high missing temperature data:", missing_data_anomalies)

# 2. High Uncertainty Regions
high_uncertainty_regions = detect_high_uncertainty_regions(df)
print("\nCities/Countries with high temperature uncertainty:", high_uncertainty_regions)

# 3. Temperature Spikes/Dips (Example for one city)
city_to_analyze = 'Chicago' #Choose a city
temperature_spikes = detect_temperature_spikes(df, city_to_analyze)
print(f"\nTemperature spikes/dips in {city_to_analyze}:\n", temperature_spikes)

# 4. Unusual Seasonal Patterns (Example for one city)
unusual_seasons = detect_unusual_seasonal_patterns(df, city_to_analyze)
print(f"\nUnusual seasonal patterns in {city_to_analyze}:\n", unusual_seasons)

# 5. Correlation-Based Anomalies
correlation_anomalies = detect_correlation_anomalies(df)
print("\nCorrelation-based temperature anomalies:\n", correlation_anomalies)

# --- Visualization (Optional) ---
# Example: Plot temperature data for Chicago with moving average and thresholds
if not temperature_spikes.empty: #Check if there are spikes to plot
    city_data = df[df['City'] == city_to_analyze].sort_values('Date').set_index('Date')
    city_data['MovingAverage'] = city_data['AverageTemperatureFahr'].rolling(window=30, center=True).mean()
    city_data['StdDev'] = city_data['AverageTemperatureFahr'].rolling(window=30, center=True).std()
    city_data['UpperThreshold'] = city_data['MovingAverage'] + 3 * city_data['StdDev']
    city_data['LowerThreshold'] = city_data['MovingAverage'] - 3 * city_data['StdDev']

    plt.figure(figsize=(12, 6))
    plt.plot(city_data['AverageTemperatureFahr'], label='Temperature')
    plt.plot(city_data['MovingAverage'], label='Moving Average')
    plt.plot(city_data['UpperThreshold'], label='Upper Threshold', linestyle='--')
    plt.plot(city_data['LowerThreshold'], label='Lower Threshold', linestyle='--')
    plt.scatter(temperature_spikes['Date'], temperature_spikes['AverageTemperatureFahr'], color='red', label='Spikes/Dips')
    plt.title(f'Temperature Anomalies in {city_to_analyze}')
    plt.xlabel('Date')
    plt.ylabel('Temperature (Fahrenheit)')
    plt.legend()
    plt.show()

# Example: Scatter plot of Latitude vs Temperature, highlighting correlation anomalies
if not correlation_anomalies.empty: #Check if there are correlation anomalies to plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['LatitudeNumeric'], df['AverageTemperatureFahr'], label='Data Points', alpha=0.5)
    plt.scatter(correlation_anomalies['LatitudeNumeric'], correlation_anomalies['AverageTemperatureFahr'], color='red', label='Anomalies')
    plt.plot(df['LatitudeNumeric'], df['ExpectedTemperature'], color='green', label='Regression Line')
    plt.xlabel('Latitude')
    plt.ylabel('Average Temperature (Fahrenheit)')
    plt.title('Latitude vs. Temperature with Correlation Anomalies')
    plt.legend()
    plt.grid(True)
    plt.show()
