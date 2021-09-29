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



# Open dataframe in new tab
from IPython.display import HTML
def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))