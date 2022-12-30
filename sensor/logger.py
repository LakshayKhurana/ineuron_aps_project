import logging
import os
from datetime import datetime


# creating log file based on the date-time stamp

LOG_FILE_NAME = f"{datetime.now().strftime('%M%D%Y__%H%M%S')}.log"


LOG_FILE_DIR = os.path.join(os.getcwd(),'logs')

#create folder if not available
os.makedirs(LOG_FILE_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
print('Printing log file path')
print(LOG_FILE_PATH)
logging.basicConfig(filename=LOG_FILE_PATH,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO,)