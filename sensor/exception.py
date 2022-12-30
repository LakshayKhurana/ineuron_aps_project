import sys
import os

class SensorException(Exception):

    def error_message_detail(error, error_detail:sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = 'Error occured in the python script [{}] at line number [{}], error message [{}] '.format(
                         file_name,exc_tb.tb_lineno, str(error))
        return error_message
    
    def __init__(self,error_message,error_detail:sys):
        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message