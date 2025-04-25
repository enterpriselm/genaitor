
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats  # Import for Shapiro-Wilk test


# --- Evaluate Residuals of ARIMA Model ---

def evaluate_arima_residuals(model_fit):
    """Evaluates the residuals of a fitted ARIMA model.

    Args:
        model_fit: The fitted ARIMA model object.
    """
    if model_fit is None:
        print("ARIMA model is None, cannot evaluate residuals.")
        return

    residuals = pd.DataFrame(model_fit.resid)
    print("\nARIMA Residual Diagnostics:")

    # 1. Summary Statistics
    print("\nResidual Summary Statistics:")
    print(residuals.describe())

    # 2. Plot Residuals over Time
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    residuals.plot(title="Residuals")
    plt.xlabel("Time")
    plt.ylabel("Residual Value")

    # 3. Density Plot
    plt.subplot(1, 2, 2)
    residuals.plot(kind='kde', title='Density')  # Use 'kde' for kernel density estimate
    plt.xlabel("Residual Value")
    plt.show()


    # 4. Autocorrelation Function (ACF) Plot
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=50, title="ACF of Residuals")  # Adjust lags as needed
    plt.show()


    # 5. Q-Q Plot (Quantile-Quantile Plot)
    import statsmodels.api as sm
    sm.qqplot(residuals.iloc[:,0], line='s') # Select the first column of the DataFrame
    plt.title("Q-Q Plot of Residuals")
    plt.show()


    # 6. Shapiro-Wilk Test for Normality
    shapiro_test = stats.shapiro(residuals)
    print("\nShapiro-Wilk Test for Normality:")
    print(f"Statistic: {shapiro_test[0]}, p-value: {shapiro_test[1]}")
    #Interpretation: Similar to Shapiro-Wilk, a low p-value suggests non-normality.


# Example Usage (after fitting the ARIMA model):

# Load the data (replace with your actual data loading)
data = pd.read_csv('examples/files/temperature.csv')
# Data Preprocessing for ARIMA example (simplified for demonstration)
data['dt'] = pd.to_datetime(data[['day', 'month', 'year']])
data.set_index('dt', inplace=True)
country = 'Poland' #Example City

city_data = data[data['Country'] == country]['AverageTemperatureFahr'].resample('M').mean().dropna()  # Monthly average, handling missing values

# Split into training and testing
train_size = int(len(city_data) * 0.8)
train, test = city_data[0:train_size], city_data[train_size:len(city_data)]

# Fit ARIMA model (example parameters - needs proper selection)
try:
    model = ARIMA(train, order=(5,1,0)) # Example parameters, you need to determine the best ones
    model_fit = model.fit()

    # Evaluate residuals
    evaluate_arima_residuals(model_fit)

    # Make predictions (example)
    predictions = model_fit.forecast(steps=len(test))

    # Evaluate predictions (example)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print(f"ARIMA RMSE: {rmse}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Training Data')
    plt.plot(test, label='Testing Data')
    plt.plot(test.index, predictions, label='ARIMA Predictions')
    plt.legend()
    plt.title(f"ARIMA Model for {country}")
    plt.show()


except Exception as e:
    print(f"Error during ARIMA model fitting or evaluation: {e}")
    model_fit = None # ensure model_fit is None if there's an error.
    # Optionally handle the error further, e.g., try different parameters.

