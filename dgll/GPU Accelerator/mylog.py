import logging

def get_logger(file_path=None):
    if file_path:
        logging.basicConfig(
            format='%(asctime)-15s %(message)s',
            level=logging.INFO,
            filename=file_path,
            filemode='w'
        )
        print("Logs are being recorded at: {}".format(file_path))
    else:
        logging.basicConfig(
            format='%(asctime)-15s %(message)s',
            level=logging.CRITICAL
        )
    log = logging.getLogger(__name__).critical
    return log
