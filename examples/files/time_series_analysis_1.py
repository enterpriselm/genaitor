
# ARIMA Parameter Selection (Example using pmdarima)
# Requires: pip install pmdarima
import pmdarima as pm

def find_best_arima_order(city_name, data):
    """Finds the best ARIMA order (p, d, q) using auto_arima."""
    city_data = data[data['City'] == city_name].copy()
    city_data = city_data.sort_values(by=['year', 'month', 'day'])
    city_data.set_index(pd.to_datetime(city_data[['year', 'month', 'day']]), inplace=True)
    city_data = city_data['AverageTemperatureFahr'].dropna()

    try:
        model = pm.auto_arima(city_data,
                              start_p=0, start_q=0,
                              max_p=5, max_q=5,
                              m=12,  # Seasonal data (12 months)
                              d=None,           # let model determine 'd'
                              seasonal=True,   # Seasonality
                              start_P=0,
                              D=None,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        return model.order, model.seasonal_order
    except Exception as e:
        print(f"Auto ARIMA failed for {city_name}: {e}")
        return None, None


# Example Usage:
city_to_model = 'Auckland'
arima_order, seasonal_order = find_best_arima_order(city_to_model, df)

if arima_order:
    print(f"Best ARIMA order for {city_to_model}: {arima_order}, Seasonal Order: {seasonal_order}")
    # Use the found order to fit the ARIMA model
    arima_model = fit_arima_for_city(city_to_model, df, order=arima_order) # Adapt fit_arima_for_city to accept seasonal_order if needed.
else:
    print("Could not determine best ARIMA order.")


# Evaluation Metrics (Example)
from sklearn.metrics import r2_score

y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)  # Add R-squared
print(f'Random Forest RMSE: {rmse_rf}')
print(f'Random Forest R-squared: {r2_rf}') # Print R-squared
