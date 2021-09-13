# Translation
import deep_translator
from deep_translator import GoogleTranslator

# Define decorator for error handler in Google translator
def sleep_retry(timeout, retry=3):
    """
    Input:
    - timeout: seconds to pause execution until retry
    - retry: number of retries (default: retry=3)
    
    Output:
    - no output
    
    """

    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                try:
                    value = function(*args, **kwargs)
                    if value is None:
                        return
                    else:
                        return value
                except:
                    print(f'Sleeping for {timeout} seconds')
                    time.sleep(timeout)
                    retries += 1
        return wrapper
    return the_real_decorator


@sleep_retry(3)
def translation_handler(x):
    """
    Input:
    - x: Any string (single word, sentence, multiple sentences).
    
    Output:
    - res: if x is German then it will be translated to English using Google translator otherwise x will be returned as is
    
    """
        
    try:
        res = GoogleTranslator(source='auto', target='en').translate(
            x[0:4999])  # Google translate can handle at most 5,000 characters.
    except KeyError:  # in case of KeyError source and target language are the same resulting in a key error. Then simply skip the translation.
        res = x
    except deep_translator.exceptions.RequestError:
        print('Request has not been successful!')
        raise deep_translator.exceptions.RequestError
        
    return res