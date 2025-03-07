
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Assuming the data is in a CSV file named 'your_data.csv'
try:
    df = pd.read_csv('examples/files/temperature.csv')
except FileNotFoundError:
    print("Error: 'temperature.csv' not found.  Make sure the file is in the correct directory, or update the file path.")
    exit()


# --- Data Quality Checks and Cleaning ---

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check if 'day' column only contains the value 1
if (df['day'] != 1).any():
    print("\nWARNING: The 'day' column contains values other than 1. Investigate.")
else:
    print("\nThe 'day' column only contains the value 1.")

# --- Latitude and Longitude Parsing ---

def parse_coordinates(coord_str):
    """Parses latitude or longitude string (e.g., '36.17S') into a float."""
    if isinstance(coord_str, str):  # Handle potential NaN values
        value = float(coord_str[:-1])
        direction = coord_str[-1]
        if direction in ['S', 'W']:
            value = -value
        return value
    else:
        return np.nan  # Return NaN if the value is not a string (e.g., NaN)


df['Latitude_parsed'] = df['Latitude'].apply(parse_coordinates)
df['Longitude_parsed'] = df['Longitude'].apply(parse_coordinates)

# Validate parsed latitude and longitude
valid_latitude = df[(df['Latitude_parsed'] >= -90) & (df['Latitude_parsed'] <= 90)]
valid_longitude = df[(df['Longitude_parsed'] >= -180) & (df['Longitude_parsed'] <= 180)]

if len(valid_latitude) != len(df):
    print("\nWARNING: Invalid latitude values found (outside -90 to 90 range).")
if len(valid_longitude) != len(df):
    print("\nWARNING: Invalid longitude values found (outside -180 to 180 range).")


# --- Descriptive Statistics ---

print("\nDescriptive Statistics for record_id:\n", df['record_id'].describe())
print("\nDescriptive Statistics for year:\n", df['year'].describe())
print("\nDescriptive Statistics for Latitude (Parsed):\n", df['Latitude_parsed'].describe())
print("\nDescriptive Statistics for Longitude (Parsed):\n", df['Longitude_parsed'].describe())


# --- Frequency Counts ---

print("\nFrequency Counts for Month:\n", df['month'].value_counts().sort_index())
print("\nFrequency Counts for Country:\n", df['Country'].value_counts().sort_values(ascending=False))  #Sort by frequency
print("\nFrequency Counts for Country ID:\n", df['country_id'].value_counts().sort_values(ascending=False)) #Sort by frequency


# --- Data Distribution Visualization ---

# Histograms
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['year'], kde=True)
plt.title('Distribution of Years')

plt.subplot(1, 3, 2)
sns.histplot(df['Latitude_parsed'], kde=True)
plt.title('Distribution of Latitude')

plt.subplot(1, 3, 3)
sns.histplot(df['Longitude_parsed'], kde=True)
plt.title('Distribution of Longitude')

plt.tight_layout()
plt.show()


# Box Plots
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.boxplot(x=df['year'])
plt.title('Boxplot of Years')

plt.subplot(1, 3, 2)
sns.boxplot(x=df['Latitude_parsed'])
plt.title('Boxplot of Latitude')

plt.subplot(1, 3, 3)
sns.boxplot(x=df['Longitude_parsed'])
plt.title('Boxplot of Longitude')

plt.tight_layout()
plt.show()


# --- Time Series Analysis (Basic) ---

# Aggregate data by year
records_per_year = df['year'].value_counts().sort_index()

# Plotting records per year
plt.figure(figsize=(12, 6))
records_per_year.plot(kind='line')
plt.title('Number of Records per Year')
plt.xlabel('Year')
plt.ylabel('Number of Records')
plt.grid(True)
plt.show()

# --- Correlation Analysis ---

correlation_matrix = df[['Latitude_parsed', 'Longitude_parsed']].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# --- Country Consistency Check ---
country_consistency = df.groupby('country_id')['Country'].nunique()
inconsistent_countries = country_consistency[country_consistency > 1]

if not inconsistent_countries.empty:
    print("\nWARNING: Inconsistent country names found for the following country IDs:")
    print(inconsistent_countries)
else:
    print("\nCountry names are consistent across country IDs.")

# --- Anomaly Detection (Basic - Record Counts per Year) ---
# Identify years with significantly lower record counts

mean_records = records_per_year.mean()
std_records = records_per_year.std()
threshold = mean_records - 2 * std_records  # Define a threshold (e.g., 2 standard deviations below the mean)

anomalous_years = records_per_year[records_per_year < threshold]

if not anomalous_years.empty:
    print("\nPotentially Anomalous Years (Low Record Counts):\n", anomalous_years)
else:
    print("\nNo significantly anomalous years found based on record counts.")


# --- Geospatial Analysis Preparation (Example) ---
# Requires a library like geopandas.  This is just a placeholder.
# import geopandas
# from shapely.geometry import Point

# # Create a GeoDataFrame
# geometry = [Point(xy) for xy in zip(df['Longitude_parsed'], df['Latitude_parsed'])]
# gdf = geopandas.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326") # WGS 84 coordinate system

# # Now you can perform spatial operations with geopandas.
# # For example, plotting the points on a map:
# # gdf.plot()
# # plt.show()
