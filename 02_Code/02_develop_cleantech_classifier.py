# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import pickle as pkl
from pyprojroot import here

# Encoding issue here!
df_train = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))

df_train.shape

df_train.head(3)

df_train.Y02.value_counts()

# There more than 500,000 non-cleantech patents and more than 40,000 cleantech patents. Use these as training data for text classification model.

df_train = df_train[['Y02', 'ABSTRACT']]

df_train = df_train.loc[df_train.ABSTRACT.notnull(),:]

# # Transformer language model 

# Note that most transformer models can only sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences (exceptions are *Longformer* and *LED*). Thus look at number of subword tokens in abstracts.

# +
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# -

sequence = "Hello it's me!"
tokens = tokenizer.tokenize(sequence)
tokens

# %%time
df_train.ABSTRACT.sample(10000).apply(lambda abstract: len(tokenizer.tokenize(abstract))).plot(kind='hist')

# Some outliers seem to exits. But the majority of abstracts is well below 1000 subword tokens. Transformer model is capable to handle different sequence length including the max sequence length allowed by the underlying model.

# Let us give it a shot and do the classification with a pretrained model without finetuning.

# +
# #!pip install tensorflow
# -

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# +
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)
