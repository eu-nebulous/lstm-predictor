import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import configparser

def find_smape(actual, forecast):
    return np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))) * 100

def create_dataset(data, timesteps):
    x, y = [], []
    for i in range(len(data) - timesteps):
        x.append(data[i:i + timesteps, 0])  # Use only the first feature (column 0)
        y.append(data[i + timesteps, 0])    # Predict the same feature
    x = np.array(x).reshape(-1, timesteps, 1)  # Reshape to (samples, timesteps, 1)
    y = np.array(y)
    return x, y

def predict_with_lstm(data_filename, attribute, next_prediction_time=None):
    # Load configuration properties
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the ini file
    config_path = os.path.join(script_dir, 'lstm.ini')

    config = configparser.ConfigParser()
    config.read(config_path)

    generate_prediction_png_output = config.getboolean('DEFAULT', 'generate_prediction_png_output')
    png_output_file = config.get('DEFAULT', 'png_output_file')

    # Load and sanitize data
    data_to_process = pd.read_csv(data_filename)
    print("Initial data loaded:")
    print(data_to_process.head())

    current_time = int(pd.Timestamp.now().timestamp())
    oldest_acceptable_time_point = current_time - (
            config.getint('DEFAULT', 'number_of_days_to_use_data_from') * 24 * 3600
            + config.getint('DEFAULT', 'prediction_processing_time_safety_margin_seconds'))

    data_to_process = data_to_process[data_to_process['ems_time'] > oldest_acceptable_time_point]
    print("Data after filtering by time:")
    print(data_to_process.head())

    if len(data_to_process) == 0:
        print("No valid data points remained after filtering. Exiting.")
        return None

    # Drop the Timestamp column to avoid it being used in predictions
    data_to_process.drop(columns=['Timestamp'], inplace=True)

    data_to_process.set_index('ems_time', inplace=True)
    data_to_process.index = pd.to_datetime(data_to_process.index, unit='s')
    data_to_process = data_to_process.resample('1s').mean().interpolate()  # 1-second intervals
    print("Data after resampling and interpolation:")
    print(data_to_process.head())

    # Prepare dataset
    forecasting_horizon = config.getint('DEFAULT', 'horizon')
    if forecasting_horizon > 0 and next_prediction_time:
        next_prediction_time = int(next_prediction_time)  # Convert to integer
        last_timestamp_data = next_prediction_time - forecasting_horizon
    else:
        last_timestamp_data = data_to_process.index[-1].timestamp()

    train_size = int(len(data_to_process) * 0.8)
    train_data = data_to_process.iloc[:train_size]
    test_data = data_to_process.iloc[train_size:]

    print("Training data sample:")
    print(train_data.head())
    print("Testing data sample:")
    print(test_data.head())

    # Scale the data (only 'cpu_usage')
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[['cpu_usage']])  # Only scale the 'cpu_usage' column
    test_scaled = scaler.transform(test_data[['cpu_usage']])

    # Prepare data for LSTM (samples, timesteps, features)
    timesteps = 10

    x_train, y_train = create_dataset(train_scaled, timesteps)
    x_test, y_test = create_dataset(test_scaled, timesteps)

    print(f"x_train shape after reshaping: {x_train.shape}")
    print(f"x_test shape after reshaping: {x_test.shape}")

    # Check the content of x_train and y_train
    print("Sample x_train data:")
    print(x_train[:5])
    print("Sample y_train data:")
    print(y_train[:5])

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(timesteps, 1)))  # Adjusted for single feature
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)

    # Make predictions
    predictions = model.predict(x_test)
    print("Raw predictions (before inverse scaling):")
    print(predictions[:5])

    predictions = scaler.inverse_transform(predictions)  # Inverse scale to get original values
    print("Predictions after inverse scaling:")
    print(predictions[:5])

    # Rescale y_test back to original scale
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    print("Actual y_test values after inverse scaling:")
    print(y_test[:5])

    # Calculate accuracy measures
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    smape = find_smape(y_test, predictions)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"MAPE: {mape}")
    print(f"SMAPE: {smape}")

    # Plot predictions
    if generate_prediction_png_output:
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, color='blue', label='Actual Data')
        plt.plot(predictions, color='red', label='LSTM Predictions')
        plt.title('LSTM Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel(attribute)
        plt.legend()
        plt.savefig(png_output_file, dpi=1200)
        plt.show()

    # Assuming predictions have been made but confidence interval is not computed
    results = {
        "prediction_value": predictions[-1][0],
        "confidence_interval": None,
        "mae": mae,
        "mse": mse,
        "mape": mape,
        "smape": smape
    }

    return results
