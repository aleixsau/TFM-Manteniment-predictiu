import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

data = pd.read_csv('dataset_v3_lstm.csv')

data = data.drop(columns=['Timestamp','Temperature','Current_RMS','Power'])  # Eliminar les columnes Timestamp, Temperature, Current_RMS i Power ja que s'ha comprovat que el model és més precís sense aquestes variables.

# Separar la columna 'label' de les dades
labels = data['label']  # Guardem els valors de la columna 'label'
data = data.drop(columns=['label'])  # Eliminem la columna 'label' del dataset

# Visualitzar les primeres files per comprovar els canvis
data.head()

# Escalar les dades per estar entre 0 i 1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Definir els passos
TIME_STEPS = 30

# Funció per crear seqüències amb una seqüencia temporal
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Crear les seqüències a partir de les dades escalades
X_train = create_sequences(data_scaled, TIME_STEPS)

# Comprovar la forma de les dades
print("Shape of training data:", X_train.shape)

from tensorflow.keras.layers import Dropout


model = Sequential([
    LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),  # Dropout per evitar el sobreajustament.
    RepeatVector(X_train.shape[1]),
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(X_train.shape[2]))
])

#model = Sequential([

#    LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
#    Dropout(0.2),  # Dropout del 20%
#    LSTM(64, activation='relu', return_sequences=True),
#    Dropout(0.2),
#    LSTM(32, activation='relu', return_sequences=False),
#    Dropout(0.2),
#    RepeatVector(X_train.shape[1]),
#    LSTM(32, activation='relu', return_sequences=True),
#    Dropout(0.2),
#    LSTM(64, activation='relu', return_sequences=True),
#    Dropout(0.2),
#    LSTM(128, activation='relu', return_sequences=True),
#    Dropout(0.2),
#    TimeDistributed(Dense(X_train.shape[2]))
#])

# Compilar el model
model.compile(optimizer='adam', loss='mse')

# Entrenar el model
history = model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.2)

# Visualitzar la pèrdua d'entrenament i validació
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Fer prediccions (reconstruir les seqüències)
X_pred = model.predict(X_train)

# Calcular l'error de reconstrucció
reconstruction_error = np.mean(np.abs(X_pred - X_train), axis=1)

# Definir un llindar per detectar anomalies
threshold = 0.1 #np.percentile(reconstruction_error, 90)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Les etiquetes originals de les seqüències (sense la finestra temporal inicial)
labels_actuals = labels[TIME_STEPS:]

# Detectar les anomalies comparant amb el llindar
# Convertim anomalies a un array 1D
anomalies = (reconstruction_error > threshold).any(axis=1)

# Matriu de confusió
cm = confusion_matrix(labels_actuals, anomalies)

# Visualització de la matriu de confusió
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

print (threshold)

# Crear seqüències amb les dades completes
X_all = create_sequences(data_scaled, TIME_STEPS)

# Predir les dades completes amb el model
X_all_pred = model.predict(X_all)

# Calcular l'error de reconstrucció
reconstruction_error = np.mean(np.abs(X_all_pred - X_all), axis=1)

# Detectar anomalies
anomalies = reconstruction_error > threshold
# Aplanar detected_anomalies si té una dimensió extra
detected_anomalies = labels[TIME_STEPS:].astype(int).values  # Aplanem les etiquetes reals si és necessari

#Detect anomalies for each feature
anomalies = anomalies.any(axis=1)

# Nombre d'anomalies detectades correctament
true_positives = np.sum(anomalies == detected_anomalies)
false_negatives = np.sum(~anomalies & detected_anomalies)
false_positives = np.sum(anomalies & ~detected_anomalies)

print(f"Nombre d'anomalies detectades correctament (true positives): {true_positives}")
print(f"False negatives: {false_negatives}")
print(f"False positives: {false_positives}")

# Gràfic de l'error de reconstrucció amb el llindar
plt.plot(reconstruction_error, label='Reconstruction error')
plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Guardar el model
model.save('model_lstm_autoencoder_v3.h5')

print("Model guardat correctament.")

from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar los datos originales de entrenamiento (esto debería ser el mismo conjunto de datos
# que utilizaste para entrenar el modelo)
data = pd.read_csv('dataset_v3_lstm.csv')

# Eliminar las columnas no relevantes (si es necesario, tal como lo hiciste en el nuevo dataset)
data = data.drop(columns=['Timestamp', 'Temperature', 'Current_RMS', 'Power', 'label'])  # Si estas columnas no son necesarias

# Escalar los datos de entrenamiento con MinMaxScaler, ya que lo usaste durante el entrenamiento
scaler = MinMaxScaler()
scaler.fit(data)  # Ajustar el escalador a los datos de entrenamiento

# Guardar el escalador para usarlo posteriormente con los nuevos datos
joblib.dump(scaler, 'scaler.save')

# Carregar el model
model = load_model('model_lstm_autoencoder_v3.h5', compile=False)

# Recompilar el model
model.compile(optimizer='adam', loss=MeanSquaredError())

print("Model carregat i recompilat correctament.")

# Carregar un nou dataset
new_data = pd.read_csv('autoencoder_dataset.csv')

# Eliminar les columnes no influents
new_data = new_data.drop(columns=['Timestamp','Temperature','Current_RMS','Power'])

# Separar la columna 'label' per a la classificació
labels_new = new_data['label']
new_data = new_data.drop(columns=['label'])

# Escalar el nou dataset utilitzant el mateix escalador
new_data_scaled = scaler.transform(new_data)

# Crear seqüències amb les noves dades
X_new = create_sequences(new_data_scaled, TIME_STEPS)

# Fer prediccions amb el model carregat
X_new_pred = model.predict(X_new)

# Calcular l'error de reconstrucció per al nou dataset
reconstruction_error_new = np.mean(np.abs(X_new_pred - X_new), axis=1)
print (reconstruction_error_new)
# Detectar anomalies en el nou dataset basades en el llindar
anomalies_new = reconstruction_error_new > 0.1

# Aplanar les etiquetes del nou dataset per tenir la mateixa mida que les seqüències processades
labels_new = labels_new[TIME_STEPS:].astype(int).values

# Detect anomalies for each feature
anomalies_new = anomalies_new.any(axis=1)

# Crear la matriu de confusió
cm = confusion_matrix(labels_new, anomalies_new)

# Visualitzar la matriu de confusió
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriu de confusió - Nou dataset")
plt.show()

# Gràfic de l'error de reconstrucció amb el llindar
plt.plot(reconstruction_error_new, label='Reconstruction error (new data)')
#plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Mostrar el nombre d'anomalies detectades en el nou dataset
print("Nombre d'anomalies detectades en el nou dataset:", np.sum(anomalies_new))

