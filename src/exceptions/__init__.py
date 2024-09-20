import os, sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,
                                                                        error_details=error_details)
        
        pass
    
    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys):
        _, _, exc_tb = error_details.exc_info()

        exception_block_line_number = exc_tb.tb_frame.f_lineno
        try_block_line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename

        error_message = f"""
        Error occured in file: {file_name}
        Line number: {try_block_line_number}
        Exception block line nubler : {exception_block_line_number}
        """
        
        logging.error(error_message)

        return error_message    

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return CustomException.__name__.str()        