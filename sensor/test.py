import sys
from sensor.exception import SensorException
try:
    a = int(input('Enter the first number = '))
    b = int(input('Enter the second number = '))
    c = a/b
    print(c)
except Exception as e:
        raise SensorException(e, sys)