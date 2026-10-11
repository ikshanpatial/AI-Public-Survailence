import serial
import time

esp32 = serial.Serial("COM5", 115200, timeout=1)
time.sleep(2)

esp32.write(b'1')
print("Buzzer ON")
time.sleep(3)

esp32.write(b'0')
print("Buzzer OFF")

esp32.close()