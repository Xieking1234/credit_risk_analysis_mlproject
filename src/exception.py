import sys 
from src.logger import logging
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename #func for filename
    error_message = "Error occured in py script name [{0}] line number [{1}] error message [{2}]".format(  #[{}] are just placeholder which are taken from the values of the function
        file_name,exc_tb.tb_lineno,str(error) #tb_lineno for err in the partiicular line no.
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)  
        self.error_message=error_message_detail(error_message,error_detail=error_detail)  

    def __str__(self):
        return self.error_message    #returned in def str cause want to return err in str format


   