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

import pandas as pd
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
            return([token.lemma_.lower() for token in doc if token.pos_ not in exclude_pos if not token.is_stop])
        else:
            return([token.lemma_.lower() for token in doc if token.pos_ not in exclude_pos])
        
    else:
        if exclude_stopwords:
            return([token.lemma_.lower() for token in doc if not token.is_stop])
        else:
            return([token.lemma_.lower() for token in doc])


# Create a color map for the distinct cleantech classes
from matplotlib import cm
import seaborn as sns
import numpy as np
temp = sns.cubehelix_palette(start=2, rot=0, dark=0.35, light=1, reverse=False, as_cmap=True).colors
ind = np.round(np.linspace(temp.shape[0]/3, temp.shape[0]-1, 8)).astype(int)
greens = temp[ind]
greens_dict = {}
for index, y02 in enumerate(['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']):
    greens_dict[y02] = greens[index]

# Function to read pickle consisting of lists
import pickle as pkl
def read_cache(cache_path):
    """Function which returns a list of lists
    
    Input: path to pickle file
        
    Output: list of lists
    
    """
    data = []
    with open(cache_path, 'rb') as f:
        try:
            while True:
                data.extend(pkl.load(f))
        except EOFError:
            pass
    return(data)

# Function which joins list of strings to one joint string while treating missing values consistently
import pandas as pd
def create_joint_string(x, columns = ['SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW']):
    return(' '.join([i for i in list(x[columns].values) if not pd.isnull(i)]))  

# Function that calculates relative count frequencies based on the Counter method.
def counter_to_relative(counter):
    total_count = sum(counter.values())
    relative = {}
    for key in counter:
        relative[key] = counter[key] / total_count
    return relative


# Calculate cosine similarity between two vectors
import numpy as np
def cosine_similarity_vectors(v1, v2):
    numerator=np.dot(v1, v2)
    denumerator1 = np.sqrt(np.sum(np.square(v1)))
    denumerator2 = np.sqrt(np.sum(np.square(v2)))
    return(numerator*1/(denumerator1*denumerator2))


# Large BERT model trained on patent data
from transformers import AutoTokenizer, AutoModel
import torch

patentbert_model = AutoModel.from_pretrained("anferico/bert-for-patents",
                                  output_hidden_states=True)
patentbert_tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")

import numpy as np

def get_word_indeces(tokenizer, text, word):
    '''
    Determines the index or indeces of the tokens corresponding to `word`
    within `text`. `word` can consist of multiple words, e.g., "cell biology".
    
    Determining the indeces is tricky because words can be broken into multiple
    tokens. I've solved this with a rather roundabout approach--I replace `word`
    with the correct number of `[MASK]` tokens, and then find these in the 
    tokenized result. 
    '''
    # Tokenize the 'word'--it may be broken into multiple tokens or subwords.
    word_tokens = tokenizer.tokenize(word)

    # Create a sequence of `[MASK]` tokens to put in place of `word`.
    masks_str = ' '.join(['[MASK]']*len(word_tokens))

    # Replace the word with mask tokens.
    s = word
    text_masked = re.sub(r'\b%s\b' % re.escape(s), masks_str, text)

    # `encode` performs multiple functions:
    #   1. Tokenizes the text
    #   2. Maps the tokens to their IDs
    #   3. Adds the special [CLS] and [SEP] tokens.
    input_ids = tokenizer.encode(text_masked)

    # Use numpy's `where` function to find all indeces of the [MASK] token.
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces


def get_embedding(b_model, b_tokenizer, text, word='', extract_CLS=True):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    extract_CLS = True: Extract CLS token as sequence embedding, otherwise take average over all contextualized word embeddings for the sequence embedding
    '''
    
    # If a word is provided, figure out which tokens correspond to it.
    if not word == '':
        word_indeces = get_word_indeces(b_tokenizer, text, word)

    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors.
    encoded_dict = b_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',     # Return pytorch tensors.
                )

    input_ids = encoded_dict['input_ids']
    
    b_model.eval()

    # Run the text through the model and get the hidden states.
    bert_outputs = b_model(input_ids)
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = b_model(input_ids)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # because we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [25 x 1 x <sentence length> x 1024]

    # Select the embeddings from the second to last layer. Other strategies are viable.
    # See here: https://jalammar.github.io/illustrated-bert/
    # `token_vecs` is a tensor with shape [<sent length> x 1024]
    token_vecs = hidden_states[-2][0]

    # either extract CLS token
    if extract_CLS:
        sentence_embedding = token_vecs[0]
    # or calculate the average of all token vectors.
    else:
        sentence_embedding = torch.mean(token_vecs, dim=0)

    # Convert to numpy array.
    sentence_embedding = sentence_embedding.detach().numpy()

    # If `word` was provided, compute an embedding for those tokens.
    if not word == '':
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)

        # Convert to numpy array.
        word_embedding = word_embedding.detach().numpy()
    
        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding



def get_word_embedding(b_model, b_tokenizer, text, word=''):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    '''
    
    # Get word indices
    word_indeces = get_word_indeces(b_tokenizer, text, word)
    
    # If no index and thus no embedding exists for the word pass
    if word_indeces.size==0:
        pass
    
    # Else extract word embedding from model
    else: 

        # Encode the text, adding the (required!) special tokens, and converting to
        # PyTorch tensors.
        encoded_dict = b_tokenizer.encode_plus(
                            text,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        input_ids = encoded_dict['input_ids']

        #b_model.eval()

        # Run the text through the model and get the hidden states.
        bert_outputs = b_model(input_ids)

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
        with torch.no_grad():

            outputs = b_model(input_ids)

            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # because we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        # `hidden_states` has shape [25 x 1 x <sentence length> x 1024]

        # Select the embeddings from the second to last layer. Other strategies are viable.
        # See here: https://jalammar.github.io/illustrated-bert/
        # `token_vecs` is a tensor with shape [<sent length> x 1024]
        token_vecs = hidden_states[-2][0]
        
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)

        # Convert to numpy array.
        word_embedding = word_embedding.detach().numpy()

        return (word_embedding)


def get_contextualized_word_embeddings(word_list, b_model, b_tokenizer, extract_CLS=False):
    # Concatenate list of words with whitespaces to a string
    word_concatenation = ' '.join(str(word) for word in word_list)
    
    # Create list of contextualized word embeddings
    word_embeddings = []
    for word in word_list:
        (_, word_emb) = get_embedding(b_model=b_model, b_tokenizer=b_tokenizer, text=word_concatenation, word=word, extract_CLS=extract_CLS)
        if isinstance(word_emb, np.ndarray):
            word_embeddings.append(list(word_emb))
    
    return(np.array(word_embeddings))



def get_contextualized_word_embeddings_sbert(keyword_list, encoder, tokenizer, output_type='embeddings_only'):
    sentence = " ".join(list(dict.fromkeys(keyword_list)))
    tokens_sen = tokenizer(sentence)['input_ids']
    token_vecs = encoder(sentence, output_value="token_embeddings")

    output = {}
    embed_vecs = []

    end = start = 1
    for i in range(len(keyword_list)):
        keyword = keyword_list[i]
        token_keyword = tokenizer.tokenize(keyword)
        start = end
        end = start + len(token_keyword)
        if keyword in output:
            keyword = keyword + "v"
        embed_vecs.append(list(token_vecs[start:end].mean(axis=0)))
        output[keyword] = {
            "tokens": token_keyword,
            "vector_ids": tokens_sen[start:end],
            "embed_vec": torch.mean(token_vecs[start:end], dim=0)
        }

    if output_type=='embeddings_only':
        return(np.array(embed_vecs))
    else:
        return output