# Instal·lar la mateixa versió de tflite que té la raspberry
#!pip uninstall tensorflow -y
#!pip install tensorflow==2.14.0
#!pip uninstall tf-keras
#!pip install keras==2.14.0

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Carregar el dataset
df = pd.read_csv('autoencoder_dataset.csv')

# Mostrar les primeres files per comprovar que les dades son correctes
df.head()

# Eliminar les etiquetes del dataset
X = df.drop(columns=['Timestamp','label'])
y = df['label']

# Separar dades normals (label = 0)
X_normal = X[y == 0]

# Escalar les dades
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)

# Mostrar dimensions per verificar
print(f"Dades normals escalades: {X_normal_scaled.shape}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


input_dim = 9  # Nombre de característiques
latent_dim = 30  # Espai latent


input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(latent_dim, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Definir el model d'autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compilar el model
autoencoder.compile(optimizer='adam', loss='mse')

# Mostrar la informació del model
autoencoder.summary()

# Entrenar el model amb dades normals
history = autoencoder.fit(X_normal_scaled, X_normal_scaled, epochs=50, batch_size=32, validation_split=0.2, shuffle=True)


# Escalar totes les dades per la detecció d'anomalies (entrenament amb normals, però detecció amb totes)
X_scaled = scaler.transform(X)

# Reconstrucció de les dades
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

# Ajust del llindar utilitzant la desviació estàndard
x_train_pred = autoencoder.predict(X_normal_scaled)
train_mae_loss = np.mean(np.abs(x_train_pred - X_normal_scaled), axis=1)
threshold = np.max(train_mae_loss)

anomalies = mse > threshold

# Comparar amb les etiquetes reals
df['predicted_anomaly'] = anomalies

# Mostrar quantes anomalies s'han detectat amb el nou llindar
print(f"Anomalies detectades: {sum(anomalies)}")
print(threshold)


# Crear una matriu de confusió per avaluar els falsos positius i els falsos negatius
cm = confusion_matrix(df['label'], df['predicted_anomaly'])
print("Matriu de confusió:\n", cm)

# Visualització de la matriu de confusió
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Informe de classificació (precision, recall, F1-score)
print("Informe de classificació:\n", classification_report(df['label'], df['predicted_anomaly']))


# Visualitzar l'error de reconstrucció
plt.figure(figsize=(10,6))
plt.hist(mse[y == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(mse[y == 1], bins=50, alpha=0.6, label='Anomaly')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Distribució de l\'error de reconstrucció')
plt.xlabel('Error de reconstrucció')
plt.ylabel('Nombre de mostres')
plt.legend()
plt.show()

# Carregar el nou dataset per validar el model amb unes altres dades
new_df = pd.read_csv('autoencoder_testmodel_dataset.csv')

# Mostrar les primeres files per assegurar que s'ha carregat correctament
new_df.head()

# Si el dataset té una columna 'label', la separes (si no, ignora aquesta part)
if 'label' in new_df.columns:
    X_new = new_df.drop(columns=['Timestamp','label'])
else:
    X_new = new_df.drop(columns=['Timestamp']) # Drop the 'Timestamp' column

# Escalar el nou conjunt de dades usant el mateix scaler que vas entrenar abans
X_new_scaled = scaler.transform(X_new)

# Fer la reconstrucció utilitzant el model entrenat
new_reconstructions = autoencoder.predict(X_new_scaled)

# Calcular l'error de reconstrucció
new_mse = np.mean(np.power(X_new_scaled - new_reconstructions, 2), axis=1)

# Detectar anomalies utilitzant el mateix threshold que abans
new_anomalies = new_mse > threshold

# Afegir les prediccions al dataframe original
new_df['predicted_anomaly'] = new_anomalies

# Mostrar el nombre d'anomalies detectades
print(f"Anomalies detectades en el nou dataset: {sum(new_anomalies)}")

true_labels = new_df['label']  # Canvia 'true_anomaly' per 'label'

# Convertir el model de Keras a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)  
tflite_model = converter.convert()

# Guardar el model convertit com a fitxer .tflite
with open('autoencoder_modeld.tflite', 'wb') as f:
    f.write(tflite_model)
# Quantitzar el model a format enter (int8) per fer-lo més eficient
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Guardar el model quantitzat
with open('autoencoder_model_quantizedd.tflite', 'wb') as f:
    f.write(tflite_quantized_model)

import pickle

# Guardar l'escalador
with open('scalerd.pkl', 'wb') as f:
    pickle.dump(scaler, f)