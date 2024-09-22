import mpu6050
import time
import math
import board 
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import csv
from datetime import datetime
import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import requests  

# Telegram Bot API Token
TELEGRAM_BOT_TOKEN = 'xxxxx' #Introduir el token i el chat id de Telegram
CHAT_ID = 'xxxxx'

# Funció per enviar alerta a Telegram
def enviar_alerta_telegram(missatge):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": missatge}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("Missatge enviat correctament.")
    else:
        print(f"Error en enviar el missatge. Codi d'error: {response.status_code}")


i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)
mpu6050 = mpu6050.mpu6050(0x68)
data_actual = datetime.now().strftime("%d%m%y_%H%M%S")
csv_filename = f"sensor_data_{data_actual}.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Accel_X", "Accel_Y", "Accel_Z", 
                     "Gyro_X", "Gyro_Y", "Gyro_Z", "Temperature", 
                     "Current_RMS", "Power", "Anomaly", "Reconstruction_Error"])

# Llegir dades del sensor mpu6500
def read_sensor_data():
    accelerometer_data = mpu6050.get_accel_data()
    gyroscope_data = mpu6050.get_gyro_data()
    temperature = mpu6050.get_temp()
    return accelerometer_data, gyroscope_data, temperature

# Llegir dades del sensor SCT-013-005
def get_corrent():
    voltatges = []
    start_time = time.time()
    while (time.time() - start_time) < 1:  # Duracio 1 segon
        valor_adc = chan.value
        voltaje_sensor = valor_adc * (4.096 / 32767.0)  # Convertir a voltatge
        corrent = voltaje_sensor * 4   # Corrent = Voltatgesensor * (5A/1V) *s'ha ajustat canviat el valor a 4, ja que dona mesures mes precises
        voltatges.append(corrent ** 2)
        time.sleep(0.01)  # Esperar 10 ms
    
    # Compensar els quadrats dels semicercles negatius
    sumatori = sum(voltatges) * 2
    corrent_rms = math.sqrt(sumatori / len(voltatges))
    return corrent_rms

# Carregar el model de TFlite
interpreter = tflite.Interpreter(model_path="autoencoder_model.tflite")
interpreter.allocate_tensors()

# Carregar l'escalador preentrenat
with open('/home/aleix/scalerd.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir el llindar de deteccio d'anomalies
threshold = 1.68  

# Deteccio d'anomalies en temps real
def make_inference(data_point):
    
    # Convertir les dades a float32 (TFlite treballa amb float32)
    input_data = np.array([data_point], dtype=np.float32)
    # Configurar l'entrada de les dades pel model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Executar l'inferencia
    interpreter.invoke()
    # Obtenir la sortida del model (reconstuccio de les dades)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    reconstruction_error = np.mean(np.power(input_data - output_data, 2))
    
    if reconstruction_error > threshold:
        return "Anomalia", reconstruction_error
    else:
        return "Normal", reconstruction_error

def main():
    print(f"Iniciant la lectura. Presiona Ctrl+C per parar.")
    start_time = time.time()

    anomaly_count = 0  # Comptador d'anomalies consecutives

    try:
        while True: # Bucle infinit per monitoritzar sense límit de temps
            accelerometer_data, gyroscope_data, temperature = read_sensor_data()
            corrent = get_corrent() 
            potencia = corrent * 230.0  
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Crear el vector de caracteristiques(dades) [Aquest vector de dades es passarà pel model TFLite per comparar la reconstrucció del model amb les dades originals]
            data_point = [
                accelerometer_data['x'], accelerometer_data['y'], accelerometer_data['z'],
                gyroscope_data['x'], gyroscope_data['y'], gyroscope_data['z'],
                temperature, corrent, potencia
            ]

            # Normalitzar les dades amb l'escalador carregat
            data_point_scaled = scaler.transform([data_point])

            resultat, error_reconstruccio = make_inference(data_point_scaled[0])

            # Comprovar si és una anomalia
            if resultat == "Anomalia":
                anomaly_count += 1
            else:
                anomaly_count = 0  # Reset del comptador si no hi ha anomalia

            # Si detecta 10 anomalies seguides, enviar una alerta a Telegram
            if anomaly_count >= 10:
                enviar_alerta_telegram("ANOMALIA AL VENTILADOR: S'han detectat 10 anomalies consecutives!")
                anomaly_count = 0  # Reset del comptador després d'enviar l'alerta

            # Guardar les dades a l'arxiu CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, 
                                 accelerometer_data['x'], accelerometer_data['y'], accelerometer_data['z'], 
                                 gyroscope_data['x'], gyroscope_data['y'], gyroscope_data['z'], 
                                 temperature, corrent, potencia, resultat, error_reconstruccio])

            time.sleep(0.5)

        print("Lectura completada. Dades guardades a", csv_filename)
        
    except KeyboardInterrupt:
        print("\nLectura aturada per l'usuari.")

if __name__ == "__main__":
    main()
