
import pandas as pd
import numpy as np
from scipy import stats

def convert_lat_long(df):
    """Converts Latitude and Longitude columns to numerical values."""
    df['Latitude'] = df['Latitude'].str.replace('N', '').str.replace('S', '-')
    df['Longitude'] = df['Longitude'].str.replace('E', '')
    df['Latitude'] = pd.to_numeric(df['Latitude'])
    df['Longitude'] = pd.to_numeric(df['Longitude'])
    return df

def impute_missing_data(df, city, method='linear'):
    """Imputes missing temperature data using linear interpolation, or mean if interpolation fails."""
    city_data = df[df['City'] == city].copy()
    city_data.sort_values(by=['year', 'month'], inplace=True)
    
    # Linear interpolation
    city_data['AverageTemperatureFahr'].interpolate(method=method, limit_direction='both', inplace=True)
    city_data['AverageTemperatureUncertaintyFahr'].interpolate(method=method, limit_direction='both', inplace=True)

    # If any NaNs remain after interpolation, fill with the city's mean temperature.
    if city_data['AverageTemperatureFahr'].isnull().any():
        city_mean = city_data['AverageTemperatureFahr'].mean()
        city_data['AverageTemperatureFahr'].fillna(city_mean, inplace=True)

    if city_data['AverageTemperatureUncertaintyFahr'].isnull().any():
        city_mean_unc = city_data['AverageTemperatureUncertaintyFahr'].mean()
        city_data['AverageTemperatureUncertaintyFahr'].fillna(city_mean_unc, inplace=True)
    
    df.update(city_data)  # Update the original dataframe
    return df

def identify_outlier_intervals(df, city, window=12, threshold=2, min_interval_length=3):
    """Identifies outlier intervals based on rolling statistics."""
    city_data = df[df['City'] == city].copy()
    city_data.sort_values(by=['year', 'month'], inplace=True)

    # Calculate rolling mean and standard deviation
    city_data['rolling_mean'] = city_data['AverageTemperatureFahr'].rolling(window=window, center=True, min_periods=6).mean()
    city_data['rolling_std'] = city_data['AverageTemperatureFahr'].rolling(window=window, center=True, min_periods=6).std()

    # Identify outliers
    city_data['z_score'] = np.abs((city_data['AverageTemperatureFahr'] - city_data['rolling_mean']) / city_data['rolling_std'])
    city_data['is_outlier'] = city_data['z_score'] > threshold

    # Find outlier intervals
    outlier_intervals = []
    start_index = None
    for i, row in city_data.iterrows():
        if row['is_outlier'] and start_index is None:
            start_index = i
            start_year = row['year']
            start_month = row['month']
        elif not row['is_outlier'] and start_index is not None:
            end_index = i
            end_year = city_data.loc[end_index-1, 'year']
            end_month = city_data.loc[end_index-1, 'month']
            
            if end_index - start_index >= min_interval_length:
                outlier_intervals.append({
                    'city': city,
                    'start_year': start_year,
                    'start_month': start_month,
                    'end_year': end_year,
                    'end_month': end_month,
                    'mean_z_score': city_data.loc[start_index:end_index-1, 'z_score'].mean()
                })
            start_index = None  # Reset start_index

    # Handle the case where the last part of the data is an outlier interval
    if start_index is not None:
        end_year = city_data['year'].iloc[-1]
        end_month = city_data['month'].iloc[-1]
        if len(city_data) - city_data.index.get_loc(start_index) >= min_interval_length:
            outlier_intervals.append({
                'city': city,
                'start_year': start_year,
                'start_month': start_month,
                'end_year': end_year,
                'end_month': end_month,
                'mean_z_score': city_data.loc[start_index:, 'z_score'].mean()
            })

    return outlier_intervals

def grubbs_test(data, alpha=0.05):
    """
    Performs Grubbs' test for outlier detection.

    Args:
        data (pd.Series or np.array): The data to test for outliers.
        alpha (float): Significance level.

    Returns:
        tuple: (bool, float) - (True if outlier detected, Grubbs' test statistic)
    """
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data)

    if sd == 0:
        return False, 0.0  # No outlier if standard deviation is zero

    z_scores = np.abs((data - mean) / sd)
    max_z = np.max(z_scores)
    
    t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    threshold = ((n - 1) / np.sqrt(n)) * np.sqrt((t_critical**2) / (n - 2 + t_critical**2))

    is_outlier = max_z > threshold
    return is_outlier, max_z


def analyze_city(df, city, window=12, threshold=2, min_interval_length=3, grubbs_alpha=0.05):
    """Analyzes a single city for outlier intervals and performs Grubbs' test."""
    print(f"Analyzing city: {city}")

    # Impute missing data
    df = impute_missing_data(df, city)

    # Identify outlier intervals
    outlier_intervals = identify_outlier_intervals(df, city, window, threshold, min_interval_length)
    if outlier_intervals:
        print("  Outlier Intervals:")
        for interval in outlier_intervals:
            print(f"    City: {interval['city']}, Start: {interval['start_year']}-{interval['start_month']}, End: {interval['end_year']}-{interval['end_month']}, Mean Z-score: {interval['mean_z_score']:.2f}")
    else:
        print("  No outlier intervals found.")

    # Grubbs' test
    city_data = df[df['City'] == city]['AverageTemperatureFahr'].dropna() # Drop any remaining NaNs after imputation
    if len(city_data) > 2: # Grubbs test needs at least 3 data points
        is_outlier, grubbs_stat = grubbs_test(city_data, grubbs_alpha)
        if is_outlier:
            print(f"  Grubbs' test: Potential outlier detected (Grubbs statistic = {grubbs_stat:.3f})")
        else:
            print(f"  Grubbs' test: No outlier detected.")
    else:
        print("  Grubbs' test: Not enough data points for the test.")


# Main execution
if __name__ == '__main__':
    # Load the data (replace 'your_data.csv' with the actual path to your CSV file)
    file_path = 'your_data.csv'  
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.  Please provide the correct path.")
        exit()

    # Convert Latitude and Longitude to numeric
    df = convert_lat_long(df)

    # Get list of unique cities
    unique_cities = df['City'].unique()

    # Analyze each city
    for city in unique_cities:
        analyze_city(df, city)
