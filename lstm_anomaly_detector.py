import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import joblib
import csv
from collections import deque
from datetime import datetime
import mpu6050
import time
import RPi.GPIO as GPIO

mpu6050 = mpu6050.mpu6050(0x68)

# Carregar el model de TensorFlow
model = tf.keras.models.load_model('model_lstm_autoencoder_v3.h5', compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())
print("Model carregat i recompilat correctament.")

# Carregar l'escalador
scaler = joblib.load('scaler.save')

TIME_STEPS = 30  # Mateix valor que en l'entrenament
data_buffer = deque(maxlen=TIME_STEPS)

# Crear un fitxer CSV amb la data actual
data_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"lstm-autoencoder_data_{data_actual}.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Reconstruction_Error", "Anomaly"])

# Configurar els pins GPIO per als LEDs
GPIO.setmode(GPIO.BCM)  # Utilitzar la numeració BCM dels pins
LED_VERD_PIN = 17
LED_VERMELL_PIN = 27

GPIO.setup(LED_VERD_PIN, GPIO.OUT)
GPIO.setup(LED_VERMELL_PIN, GPIO.OUT)

# Inicialitzar els LEDs apagats
GPIO.output(LED_VERD_PIN, GPIO.LOW)
GPIO.output(LED_VERMELL_PIN, GPIO.LOW)

def read_sensor_data():
    accelerometer_data = mpu6050.get_accel_data()
    gyroscope_data = mpu6050.get_gyro_data()

    # Crear el punt de dades que conté les lectures rellevants
    data_point = [
        accelerometer_data['x'], accelerometer_data['y'], accelerometer_data['z'],
        gyroscope_data['x'], gyroscope_data['y'], gyroscope_data['z']
    ]

    return data_point

# Funció per crear una seqüència a partir de les últimes dades
def create_sequence(data_buffer):
    return np.array(data_buffer)

anomaly_threshold = 0.1

# Bucle principal per llegir les dades en temps real
try:
    print("Iniciant la monitorització en temps real. Premeu Ctrl+C per aturar.")
    while True:
        data_point = read_sensor_data()
        # Afegir el punt de dades al buffer
        data_buffer.append(data_point)
        
        # Només processa si tenim prou dades (TIME_STEPS)
        if len(data_buffer) == TIME_STEPS:
            # Convertir el buffer en una seqüència
            sequence = create_sequence(data_buffer)
            # Escalar la seqüència
            sequence_scaled = scaler.transform(sequence)
            # Afegir una nova dimensió per a predicció (shape: 1, TIME_STEPS, num_features)
            sequence_scaled = np.expand_dims(sequence_scaled, axis=0)
            # Realitzar la predicció amb el model
            sequence_pred = model.predict(sequence_scaled)
            # Calcular l'error de reconstrucció
            reconstruction_error = np.mean(np.abs(sequence_pred - sequence_scaled))
            # Comprovar si l'error de reconstrucció supera el llindar
            is_anomaly = reconstruction_error > anomaly_threshold
            # Obtenir el temps actual
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Guardar les dades i el resultat al fitxer CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, data_point[0], data_point[1], data_point[2], 
                                 data_point[3], data_point[4], data_point[5], 
                                 reconstruction_error, is_anomaly])

            # Mostrar si s'ha detectat una anomalia i controlar els LEDs
            if is_anomaly:
                print(f"Anomalia detectada! Error de reconstrucció: {reconstruction_error}")
                GPIO.output(LED_VERD_PIN, GPIO.LOW)   # Apagar el LED verd
                GPIO.output(LED_VERMELL_PIN, GPIO.HIGH)  # Encendre el LED vermell
            else:
                print(f"Dades normals. Error de reconstrucció: {reconstruction_error}")
                GPIO.output(LED_VERD_PIN, GPIO.HIGH)  # Encendre el LED verd
                GPIO.output(LED_VERMELL_PIN, GPIO.LOW)  # Apagar el LED vermell

        
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\nMonitorització en temps real aturada.")

finally:
    # Apagar tots els LEDs abans de netejar
        GPIO.output(LED_VERD_PIN, GPIO.LOW)
        GPIO.output(LED_VERMELL_PIN, GPIO.LOW)
    # Netejar la configuració dels pins GPIO quan el programa finalitza
        GPIO.cleanup()
