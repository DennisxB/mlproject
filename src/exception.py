import sys
import logger
# custom exception

def error_message_detail(error, error_detail:sys):
    """
    Functionality to get detailed error information
    """
    _, _, exc_tb = error_detail.exc_info()

    # Get file name 
    file_name = exc_tb.tb_frame.f_code.co_filename
    #exc_tb gets the file and line which the error occured
    error_message = f"error occured in python script name [{file_name}] \
            line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    
    return error_message

# Creating a Custom Exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message


if __name__=="__main__":
    try:
        1 / 0
    
    except Exception as e:
        logger.logging.info("Divide by Zero")
        raise CustomException(e, error_detail=sys)
        
        
