# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import re
import spacy
import pickle as pkl
from pyprojroot import here
from tqdm import tqdm

# +
# Install a default trained English pipeline package: https://spacy.io/models/en
# #!python -m spacy download en_core_web_sm
# -

# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load("en_core_web_sm")

# Default tokenizer seperates hyphen words in three tokens e.g.:
# - photo-sensor -> ['photo', '-', 'sensor']
# - aluminum-silicon -> ['aluminium', '-', 'silicon']
#
# In the context of developing semantic spaces for different technologies this is undesirable as hyphen words possibly carry high value in terms of describing the underlying technology.
#
# Thus, the default tokenizer will be customized to convert hyphen words into a single token following the suggestion found [here](https://stackoverflow.com/questions/51012476/spacy-custom-tokenizer-to-include-only-hyphen-words-as-tokens-using-infix-regex), i.e.:
# - photo-sensor -> ['photo-sensor']

# +
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
# -

# Load patent abstracts
df = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))

df.head(3)

df.iloc[0].ABSTRACT

doc = nlp(df.iloc[1].ABSTRACT)

' '.join([token.lemma_ for token in doc])

[(token.lemma_, token.pos_, token.is_stop) for token in doc]

spacy.explain('PROPN')

spacy.explain('ADJ')

spacy.explain('X')


# Define tokenizer that returns word lemmas and skips punctuation, numbers and stop words.

def string_to_lemma(doc):
    doc = nlp(doc)
    return([token.lemma_ for token in doc if token.pos_ not in ['PUNCT', 'NUM', 'X'] if not token.is_stop])


df.iloc[0].ABSTRACT

string_to_lemma(df.iloc[0].ABSTRACT)

df_lemma = pd.DataFrame(columns=['APPLN_ID', 'LEMMA_LIST'])
temp=[]
for index, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
    doc = row.ABSTRACT
    appln_id = row.APPLN_ID
    temp.append((appln_id, string_to_lemma(doc)))
    if index == 5:
        break


pd.DataFrame(temp)
