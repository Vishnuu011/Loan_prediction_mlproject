from loan_prediction.logger import logging
from loan_prediction.exception import CustomException
import sys

logging.info("demo log")

try:
        a=1/0

except Exception as e:
    logging.info("Division by Zero")
    raise CustomException(e,sys)