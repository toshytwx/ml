import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


data = pd.read_csv("lung_cancer_data.csv")
X = data.drop(columns=["Survival_Months"])
y = data["Survival_Months"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_min, y_max = y_train.min(), y_train.max()

layer_configs = [(64, 32), (128, 64), (64, 64, 32)]
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = 100

results = []

for layers in layer_configs:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Навчання моделі: Layers: {layers}, Learning Rate: {lr}, Batch Size: {batch_size}")

            model = Sequential()
            model.add(Input(shape=(X_train.shape[1],)))  
            for layer_size in layers:
                model.add(Dense(layer_size, activation='relu'))
            model.add(Dense(1))  

            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            y_pred = model.predict(X_test).flatten()
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            nrmse = rmse / (y_max - y_min)

            results.append([layers, lr, batch_size, rmse, nrmse])

results_df = pd.DataFrame(results, columns=["Layer Configs", "Learning Rate", "Batch Size", "RMSE", "NRMSE"])
results_df.to_csv("neural_network_results.csv", index=False)
print("Результати збережено у файл 'neural_network_results.csv'")
