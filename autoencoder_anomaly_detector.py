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
import pandas as pd
import tflite_runtime.interpreter as tflite
import requests  
import pickle  
import RPi.GPIO as GPIO


# Telegram Bot API Token
TELEGRAM_BOT_TOKEN = 'XXXXX'
CHAT_ID = 'XXXX'

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
csv_filename = f"autoencoder_anomaly_detector_{data_actual}.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Accel_X", "Accel_Y", "Accel_Z", 
                     "Gyro_X", "Gyro_Y", "Gyro_Z", "Temperature", 
                     "Current_RMS", "Power", "Anomaly", "Reconstruction_Error"])

# Configurar els pins GPIO per als LEDs
GPIO.setmode(GPIO.BCM)  # Utilitzar la numeració BCM dels pins
LED_VERD_PIN = 17
LED_VERMELL_PIN = 27

GPIO.setup(LED_VERD_PIN, GPIO.OUT)
GPIO.setup(LED_VERMELL_PIN, GPIO.OUT)

# Inicialitzar els LEDs apagats
GPIO.output(LED_VERD_PIN, GPIO.LOW)
GPIO.output(LED_VERMELL_PIN, GPIO.LOW)

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
    while (time.time() - start_time) < 1:  # Duracio de 1 segon
        valor_adc = chan.value
        voltatge_sensor = valor_adc * (4.096 / 32767.0)  # Convertir a voltatge
        corrent = voltatge_sensor * 4   # Corrent = Voltatgesensor * (5A/1V) *s'ha ajustat canviat el valor a 4, ja que dona mesures mes precises
        voltatges.append(corrent ** 2)
        time.sleep(0.01)  # Esperar 10 ms
    
    sumatoria = sum(voltatges) * 2
    corrent_rms = math.sqrt(sumatoria / len(voltatges))
    return corrent_rms

# Carregar el model TFLite
interpreter = tflite.Interpreter(model_path="autoencoder_model.tflite")
interpreter.allocate_tensors()

# Carregar l'escalador des del fitxer .pkl
with open('scalerd.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Obtenir detalls dels tensors d'entrada i sortida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir el threshold per detectar anomalies
threshold = 1.68  # Ajusta el threshold segons el model

# Noms de les característiques utilitzades durant l'entrenament
feature_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Temperature', 'Current_RMS', 'Power']

def make_inference(data_point):
    """
    Aquesta funció fa inferència en temps real sobre un punt de dada normalitzat.
    """
    # Convertir les dades a float32 (el model TFLite treballa amb float32)
    input_data = np.array([data_point], dtype=np.float32)
    # Configurar l'entrada de dades per al model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Executar inferència
    interpreter.invoke()
    # Obtenir la sortida del model (reconstrucció de les dades)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Calcular l'error de reconstrucció (MAE en lloc de MSE)
    reconstruction_error = np.mean(np.abs(input_data - output_data))
 
    # Determinar si és una anomalia
    if reconstruction_error > threshold:
        return "Anomalia", reconstruction_error
    else:
        return "Normal", reconstruction_error


def main():
    print(f"Iniciant la medició. Prem Ctrl+C per aturar.")
    start_time = time.time()
    anomaly_count = 0  # Comptador d'anomalies consecutives

    try:
        while True: # Bucle infinit per monitoritzar sense límit de temps
            
            accelerometer_data, gyroscope_data, temperature = read_sensor_data()
            corrent = get_corrent()
            potencia = corrent * 230.0  # Potencia = I * V (Watts)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Crear el punt de dades per al autoencoder
            data_point = [
                accelerometer_data['x'], accelerometer_data['y'], accelerometer_data['z'],
                gyroscope_data['x'], gyroscope_data['y'], gyroscope_data['z'],
                temperature, corrent, potencia
            ]

            # Crear un DataFrame amb els mateixos noms de columnes que durant l'entrenament
            data_point_df = pd.DataFrame([data_point], columns=feature_names)

            # Normalitzar les dades amb l'escalador carregat
            data_point_scaled = scaler.transform(data_point_df)

            # Fer inferència i determinar si és una anomalia
            resultat, error_reconstruccion = make_inference(data_point_scaled[0])

            # Comprovar si és una anomalia
            if resultat == "Anomalia":
                anomaly_count += 1
                GPIO.output(LED_VERD_PIN, GPIO.LOW)   # Apagar el LED verd
                GPIO.output(LED_VERMELL_PIN, GPIO.HIGH)  # Encendre el LED vermell
            else:
                anomaly_count = 0  # Reiniciar el comptador si no hi ha anomalia
                GPIO.output(LED_VERD_PIN, GPIO.HIGH)  # Encendre el LED verd
                GPIO.output(LED_VERMELL_PIN, GPIO.LOW)  # Apagar el LED vermell

            # Si detecta 10 anomalies seguides, enviar una alerta a Telegram
            if anomaly_count >= 10:
                enviar_alerta_telegram("ANOMALIA AL VENTILADOR: S'han detectat 10 anomalies consecutives!")
                anomaly_count = 0  # Reiniciar el comptador després d'enviar l'alerta

            # Guardar les dades en el fitxer CSV, incloent si és una anomalia
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, 
                                 accelerometer_data['x'], accelerometer_data['y'], accelerometer_data['z'], 
                                 gyroscope_data['x'], gyroscope_data['y'], gyroscope_data['z'], 
                                 temperature, corrent, potencia, resultat, error_reconstruccion])

            # Esperar abans de la següent mesura
            time.sleep(1.0)

        print("Mesura completada. Dades guardades a", csv_filename)
        
    except KeyboardInterrupt:
        print("\nMesura aturada per l'usuari.")
        
    finally:
    # Apagar tots els LEDs abans de netejar
        GPIO.output(LED_VERD_PIN, GPIO.LOW)
        GPIO.output(LED_VERMELL_PIN, GPIO.LOW)
    # Netejar la configuració dels pins GPIO quan el programa finalitza
        GPIO.cleanup()

if __name__ == "__main__":
    main()

