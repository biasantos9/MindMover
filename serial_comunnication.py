from serial import Serial
import time

#if robot = true - comunnication is being made
#if robot = false - prints the values received
#values come from eegrealtime_again_trial.py 

ROBOT = True
if ROBOT:
    ser = Serial('COM10', 9600)
    last_timestamp = 0

    def send(value):
        
        mapping = {'up':0, 'down':1, 'right':2, 'left':3}
        values = mapping.get(value[0])

        global ser
        while True:
            try:
                #values are sent to the serial port
                ser.write(bytes(values))
                break
            except:
                time.sleep(0.1)
                ser.close()
                while True:
                    try:
                        ser = serial.Serial('COM10', 9600)
                    except:
                        time.sleep(0.1)
else:
    def send(value):
        mapping = {'up': 0, 'down': 1, 'right': 2, 'left': 3}
        values = mapping.get(value[0])
    
        time.sleep(0.1)
    


