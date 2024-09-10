# Este es un ejemplo de utilización de una red LSTM para predecir la serie temporal del índice
# de precios de acciones utilizando datos históricos.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Cargar datos
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
dataframe = pd.read_csv(url, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalizar datos
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Dividir datos en conjuntos de entrenamiento y prueba
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Función para crear dataset con serie temporal
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Reshape en X=t and Y=t+1
time_step = 2
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)


X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# Crear y compilar el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, time_step)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenamos el modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=64, verbose=1)

# Predicción
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transformar a la escala original
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])


# Calculamos RMSE
train_score = np.sqrt(np.mean(np.power((y_train[0] - train_predict[:,0]),2)))
print('Train RMSE:', train_score)
test_score = np.sqrt(np.mean(np.power((y_test[0] - test_predict[:,0]),2)))
print('Test RMSE:', test_score)

# Graficamos resultados
plt.figure(figsize=(10,6))
plt.plot(scaler.inverse_transform(dataset), label='Original Data')
plt.plot(np.concatenate((train_predict, test_predict)), label='Predicted Data')
plt.legend()
plt.show()






















