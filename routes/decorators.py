import time
from functools import wraps
import logging

log = logging.getLogger("Decorators")

def retry_with_exponential_backoff(max_retries=5, initial_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    log.warning(f"Attempt {i + 1}/{max_retries} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
            log.error(f"Function failed after {max_retries} retries.")
            raise Exception(f"Function failed after {max_retries} retries.")
        return wrapper
    return decorator