
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def create_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    """
    Creates an LSTM model for time series forecasting.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        lstm_units (int): Number of LSTM units in the LSTM layer.
        dropout_rate (float): Dropout rate to prevent overfitting.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tensorflow.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units, return_sequences=False))  # Last LSTM layer doesn't return sequences
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Output layer (predicting one value)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')  # Or use 'mae'
    return model



def prepare_data_for_lstm(data, city_name, time_steps=60):
    """
    Prepares the data for LSTM training.

    Args:
        data (pd.DataFrame): The dataframe containing time series data.
        city_name (str): The name of the city to filter data.
        time_steps (int): The number of time steps to use for each input sequence.

    Returns:
        tuple: A tuple containing the training data (X_train, y_train),
               testing data (X_test, y_test), and the scaler object.
    """
    city_data = data[data['City'] == city_name].copy()
    city_data = city_data.sort_values(by=['year', 'month', 'day'])
    city_data.set_index(pd.to_datetime(city_data[['year', 'month', 'day']]), inplace=True)
    city_data = city_data['AverageTemperatureFahr'].dropna()

    # Reshape to a DataFrame for scaling
    city_data = city_data.values.reshape(-1, 1)


    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(city_data)

    # Create sequences
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM input (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) #shuffle=False is critical for time series data
    return X_train, X_test, y_train, y_test, scaler


# Example Usage:

# Load the data (replace with your actual data loading)
data = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv')# Basic data cleaning
df = data.copy()
df = df.rename(columns={'dt': 'Date', 'AverageTemperature': 'AverageTemperatureCelsius'})
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['AverageTemperatureFahr'] = df['AverageTemperatureCelsius'] * 9/5 + 32
df = df.dropna(subset=['Latitude', 'Longitude', 'AverageTemperatureFahr', 'City'])  # Drop rows with NaN in essential columns
df['Latitude'] = df['Latitude'].apply(lambda x: float(x[:-1]) if isinstance(x, str) else x)
df['Longitude'] = df['Longitude'].apply(lambda x: float(x[:-1]) if isinstance(x, str) else x)

# A more sophisticated approach might involve time series imputation methods.
df['AverageTemperatureFahr'] = df.groupby('City')['AverageTemperatureFahr'].transform(lambda x: x.fillna(x.mean()))
# 2. Cyclical Encoding for 'month'
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 3. Interaction Term
df['Latitude_month'] = df['Latitude'] * df['month']


# 4. Define features
numerical_features = ['year', 'Latitude', 'Longitude', 'month_sin', 'month_cos', 'Latitude_month']
categorical_features = ['City']

# 5. Create Preprocessor
# Imputation for numerical features (handle any remaining NaNs)
# One-Hot Encoding for categorical features
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  #Impute remaining NaNs if any exist
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handle unknown categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


city_to_model = 'Auckland'  # Or any other city in your dataset
time_steps = 60

X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(df, city_to_model, time_steps)

# Define LSTM model parameters
lstm_units = 64
dropout_rate = 0.2
learning_rate = 0.001
epochs = 50
batch_size = 32


# Create and train the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
model = create_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping], verbose=1)


# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean Squared Error on the test set: {loss}')


# Make predictions
predicted_temps = model.predict(X_test)

# Inverse transform the predictions to the original scale
predicted_temps = scaler.inverse_transform(predicted_temps)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Actual Temperature')
plt.plot(predicted_temps, label='Predicted Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (Fahrenheit)')
plt.title(f'Temperature Prediction for {city_to_model}')
plt.legend()
plt.show()


# Plot training history
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('