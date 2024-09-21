import mpu6050
import time
import math
import board 
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import csv
from datetime import datetime

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)
mpu6050 = mpu6050.mpu6050(0x68)

# Preguntar a l'usuari pel mode de funcionament i la velocitat
funcionament = input("Funcionament (normal o anomalia): ").lower()
velocitat = input("Velocitat (1, 2 o 3): ")

# Validar l'entrada de l'usuari
if funcionament not in ["normal", "anomalia"]:
    print("Mode de fucionament incorrecte. Ha de ser 'normal' o 'anomalia'.")
    exit(1)

if velocitat not in ["1", "2", "3"]:
    print("Velocitat no valida. Ha de ser '1', '2' o '3'.")
    exit(1)

data_actual = datetime.now().strftime("%d%m%y")

csv_filename = f"sensor_data_{funcionament}_velocidad_{velocitat}__{data_actual}.csv"

# Crear l'arxiu CSV i escriure el nom de les variables
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Accel_X", "Accel_Y", "Accel_Z", 
                     "Gyro_X", "Gyro_Y", "Gyro_Z", "Temperature", 
                     "Current_RMS", "Power"])

# Llegir dades del sensor mpu6500
def read_sensor_data():
    accelerometer_data = mpu6050.get_accel_data()
    gyroscope_data = mpu6050.get_gyro_data()
    temperature = mpu6050.get_temp()
    return accelerometer_data, gyroscope_data, temperature

# Calcular la corrent 
def get_corriente():
    voltatges = []
    start_time = time.time()
    while (time.time() - start_time) < 1:  # Duracio 1 segon
        valor_adc = chan.value
        voltatge_sensor = valor_adc * (4.096 / 32767.0)  # Convertir a voltatge
        corrent = voltatge_sensor * 4   # Corrent = Voltatgesensor * (5A/1V) *s'ha ajustat canviat el valor a 4, ja que dona mesures mes precises
        voltatges.append(corrent ** 2)
        time.sleep(0.01)  # Esperar 10 ms
    
    # Compensar els quadrats dels semicercles negatius
    sumatori = sum(voltatges) * 2
    corrent_rms = math.sqrt(sumatori / len(voltatges))
    return corrent_rms

def main():
    print(f"Iniciant la mesura durant 1 hora en mode {funcionament} i velocitat {velocitat}. Prem Ctrl+C per aturar.")
    start_time = time.time()
    try:
        while (time.time() - start_time) < 3600:  # 3600 segons
            # Llegir dades del sensor MPU6050
            accelerometer_data, gyroscope_data, temperature = read_sensor_data()
            print("Accelerometer data:", accelerometer_data)
            print("Gyroscope data:", gyroscope_data)
            print("Temp:", temperature)

            # Llegir i calcular el corrent i la potencia
            corrent = get_corriente() 
            potencia = corrent * 230.0  # Potencia = I * V (Watts)
            print(f"Irms: {corrent:.3f} A, Potencia: {potencia:.3f} W")
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Guardar les dades
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, 
                                 accelerometer_data['x'], accelerometer_data['y'], accelerometer_data['z'], 
                                 gyroscope_data['x'], gyroscope_data['y'], gyroscope_data['z'], 
                                 temperature, corrent, potencia])

            # Esperar 0,5 segons abans de la seguent mesura
            time.sleep(0.5)

        print("Mesura completada. Dades guardades a", csv_filename)
        
    except KeyboardInterrupt:
        print("\nMesura aturada per l'usuari.")

if __name__ == "__main__":
    main()
