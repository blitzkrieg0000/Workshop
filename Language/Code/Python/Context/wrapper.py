# WRAPPERS

from functools import wraps
import time
import logging

def my_logger(orig_func):

    logging.basicConfig(filename=f'context/files/{orig_func.__name__}.log', level=logging.INFO)

    #@wraps(orig_func)
    def wrapper(*args, **kwargs):
        print(f'Ran with args: {args}, and kwargs: {kwargs}')
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)

    return wrapper

def my_timer(orig_func):

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print(f'{orig_func.__name__} ran in: {t2} sec')
        return result

    return wrapper

@my_logger
@my_timer
def display_info(name, age):
    time.sleep(1)
    print(f'display_info ran with arguments ({name}, {age})')

X = display_info('Merhaba', 100)
print(X)