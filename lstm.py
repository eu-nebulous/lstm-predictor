import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import layers, models, regularizers
from geneticalgorithm2 import geneticalgorithm2 as ga
import itertools
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO)

# Step 1: Data preprocessing
data = pd.read_csv('~/Pobrane/usagedata.csv')
data_cleaned = data.dropna()

# Ensure datetime columns are not included in numerical operations
date_columns = ['timestamp']  # Adjust this if you have other non-numeric columns
numeric_columns = data_cleaned.columns.difference(date_columns)

# Data augmentation function
def augment_data(df, noise_level=0.01):
    augmented_data = df.copy()
    noise = np.random.normal(0, noise_level, df.shape)
    augmented_data += noise
    return augmented_data

augmented_data = augment_data(data_cleaned[numeric_columns], noise_level=0.05)
data_augmented = pd.concat([data_cleaned[numeric_columns], augmented_data], ignore_index=True)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_augmented)
X = data_scaled[:, :-1]
y = data_scaled[:, -1]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape the data to add a third dimension
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 3: LSTM with regularization
def build_lstm(param1, param2, param3, dropout_rate):
    units = max(1, int(param1 * 64))  # Ensure units is at least 1
    num_layers = max(1, int(param2 * 5))  # Ensure num_layers is at least 1
    
    input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_layer = input_layer

    for _ in range(num_layers):
        lstm_layer = layers.LSTM(units=units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(lstm_layer)
        lstm_layer = layers.ReLU()(lstm_layer)
    
    lstm_layer = layers.GlobalAveragePooling1D()(lstm_layer)
    lstm_layer = layers.Dropout(dropout_rate)(lstm_layer)
    output_layer = layers.Dense(1, activation='linear')(lstm_layer)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Step 2: Genetic Algorithms (GA)
def fitness_function(params):
    try:
        param1, param2, param3, dropout_rate = params
        logging.info(f"Running fitness_function with params: {params}")
        model = build_lstm(param1, param2, param3, dropout_rate)
        if model is None:
            logging.warning("Invalid parameters for model building.")
            return float('inf')  # Return a high loss for invalid parameters
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=0)  # Reduced epochs
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        logging.info(f"Validation loss: {val_loss}")
        return val_loss
    except Exception as e:
        logging.error(f"An error occurred in fitness_function: {e}")
        return float('inf')  # Return a high loss if an error occurs

varbound = np.array([[0, 1], [0, 1], [0, 1], [0, 0.5]])
algorithm_param = {
    'max_num_iteration': 50,  # Reduced iterations
    'population_size': 5,  # Reduced population size
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'parents_portion': 0.4,  # Ensure this value leads to a valid parents count
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

# Running the genetic algorithm

model = ga(function=None, dimension=4, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run(function=fitness_function)
best_params = model.output_dict['variable']

# Ensure best_params is a list of floats
best_params = [float(param) for param in best_params]

# Using the best parameters to build the final model
best_model = build_lstm(*best_params)
if best_model is not None:
    history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)  # Increased epochs for final training
    test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
else:
    print("The best parameters resulted in an invalid model configuration.")

# Visualize the training history
def plot_history(history, title='Model Loss'):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history, 'Model Loss - Final Training')

# Additional experiments with different hyperparameters and regularization
unit_ranges = [32, 64]
num_layers_ranges = [1, 2]
dropout_rate_ranges = [0.1, 0.3]
results = []

# Use joblib for parallel computation
def evaluate_model(units, num_layers, dropout_rate):
    def build_custom_lstm():
        input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_layer = input_layer

        for _ in range(num_layers):
            lstm_layer = layers.LSTM(units=units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(lstm_layer)
            lstm_layer = layers.ReLU()(lstm_layer)
        
        lstm_layer = layers.GlobalAveragePooling1D()(lstm_layer)
        lstm_layer = layers.Dropout(dropout_rate)(lstm_layer)
        output_layer = layers.Dense(1, activation='linear')(lstm_layer)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    model = build_custom_lstm()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=0)  # Reduced epochs
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    return (units, num_layers, dropout_rate, test_loss, test_mae)

# Parallelize the evaluation of different hyperparameters
results = Parallel(n_jobs=-1)(delayed(evaluate_model)(units, num_layers, dropout_rate)
                              for units, num_layers, dropout_rate in itertools.product(unit_ranges, num_layers_ranges, dropout_rate_ranges))

for result in results:
    units, num_layers, dropout_rate, test_loss, test_mae = result
    print(f'Units: {units}, Num Layers: {num_layers}, Dropout Rate: {dropout_rate}, Test Loss: {test_loss}, Test MAE: {test_mae}')

best_result = min(results, key=lambda x: x[3])
print(f'Best Result - Units: {best_result[0]}, Num Layers: {best_result[1]}, Dropout Rate: {best_result[2]}, Test Loss: {best_result[3]}, Test MAE: {best_result[4]}')

# Hyperparameter optimization using RandomizedSearchCV
def create_model(units=64, num_layers=1, dropout_rate=0.1):
    input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_layer = input_layer

    for _ in range(num_layers):
        lstm_layer = layers.LSTM(units=units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(lstm_layer)
        lstm_layer = layers.ReLU()(lstm_layer)
    
    lstm_layer = layers.GlobalAveragePooling1D()(lstm_layer)
    lstm_layer = layers.Dropout(dropout_rate)(lstm_layer)
    output_layer = layers.Dense(1, activation='linear')(lstm_layer)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = KerasRegressor(model=create_model, epochs=10, batch_size=32, verbose=0)
param_distributions = {
    'model__units': [32, 64],
    'model__num_layers': [1, 2],
    'model__dropout_rate': [0.1, 0.3]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=10, cv=3, verbose=1, n_jobs=-1)
random_search_result = random_search.fit(X_train, y_train)
best_params_random_search = random_search_result.best_params_
print(f'Best Hyperparameters: {best_params_random_search}')

best_model = create_model(**{k.split("__")[1]: v for k, v in best_params_random_search.items()})
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)  # Increased epochs for final training
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

plot_history(history, 'Model Loss - RandomizedSearchCV')

# Adding lag features to data
def create_lag_features(df, lag=3):
    for i in range(1, lag + 1):
        df[f'cpu_usage_lag_{i}'] = df['cpu_usage'].shift(i)
        df[f'ram_usage_lag_{i}'] = df['ram_usage'].shift(i)
        df[f'io_usage_lag_{i}'] = df['io_usage'].shift(i)
        df[f'network_usage_lag_{i}'] = df['network_usage'].shift(i)
    df = df.dropna()
    return df

data_lagged = create_lag_features(data_cleaned)

# Data preprocessing with lag features
data_scaled = scaler.fit_transform(data_lagged[numeric_columns])
X = data_scaled[:, :-1]
y = data_scaled[:, -1]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape the data to add a third dimension
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Defining the model with lag features
def build_lstm_with_lag(param1, param2, param3, dropout_rate):
    units = max(1, int(param1 * 64))  # Ensure units is at least 1
    num_layers = max(1, int(param2 * 5))  # Ensure num_layers is at least 1
    
    input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_layer = input_layer

    for _ in range(num_layers):
        lstm_layer = layers.LSTM(units=units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(lstm_layer)
        lstm_layer = layers.ReLU()(lstm_layer)
    
    lstm_layer = layers.GlobalAveragePooling1D()(lstm_layer)
    lstm_layer = layers.Dropout(dropout_rate)(lstm_layer)
    output_layer = layers.Dense(1, activation='linear')(lstm_layer)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Training and evaluating the model with lag features
best_model_with_lag = build_lstm_with_lag(*best_params)
if best_model_with_lag is not None:
    history_with_lag = best_model_with_lag.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)  # Increased epochs for final training
    test_loss, test_mae = best_model_with_lag.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss with Lag Features: {test_loss}, Test MAE with Lag Features: {test_mae}')
else:
    print("The best parameters resulted in an invalid model configuration.")

plot_history(history_with_lag, 'Model Loss with Lag Features')

# Calculate RMSE for final evaluation
test_predictions = best_model_with_lag.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'Test RMSE with Lag Features: {rmse}')

