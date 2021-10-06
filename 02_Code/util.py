# Translation
import deep_translator
import time
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


# Lemmatization
import spacy
import re

# Initialize spacy 'en' model
nlp = spacy.load("en_core_web_sm")

#Default tokenizer seperates hyphen words in three tokens e.g.:
#- photo-sensor -> ['photo', '-', 'sensor']
#- aluminum-silicon -> ['aluminium', '-', 'silicon']

#In the context of developing semantic spaces for different technologies this is undesirable as hyphen words possibly carry high value in terms of describing the #underlying technology.

#Thus, the default tokenizer will be customized to convert hyphen words into a single token following the suggestion found [here](https://stackoverflow.com/questions/51012476/spacy-custom-tokenizer-to-include-only-hyphen-words-as-tokens-using-infix-regex), i.e.:
#- photo-sensor -> ['photo-sensor']
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)
nlp.tokenizer = custom_tokenizer(nlp)

def string_to_lemma(doc, exclude_pos = ['PUNCT', 'NUM', 'X'], exclude_stopwords = True):
    """Function which returns a list of lemmatized words.
    
    Input: string representing a the textual content of a document.
    
    Arguments:
        - exclude_pos: list of part-of-speech abbreviations which will be excluded in the output. 
            See https://universaldependencies.org/docs/u/pos/ for an overview of pos tags
        - exclude_stopwords: specify whether stop words will be excluded in the output
        
    Output: list of strings representing the word lemmas from the input.
    
    """
    
    # Calling the nlp object on a string of text will return a processed Doc
    doc = nlp(doc)
    
    if exclude_pos:
        if exclude_stopwords:
            return([token.lemma_ for token in doc if token.pos_ not in exclude_pos if not token.is_stop])
        else:
            return([token.lemma_ for token in doc if token.pos_ not in exclude_pos])
        
    else:
        if exclude_stopwords:
            return([token.lemma_ for token in doc if not token.is_stop])
        else:
            return([token.lemma_ for token in doc])


# Create a color map for the distinct cleantech classes
from matplotlib import cm
temp = sns.cubehelix_palette(start=2, rot=0, dark=0.35, light=1, reverse=False, as_cmap=True).colors
ind = np.round(np.linspace(temp.shape[0]/3, temp.shape[0]-1, 8)).astype(int)
greens = temp[ind]
greens_dict = {}
for index, y02 in enumerate(['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']):
    greens_dict[y02] = greens[index]