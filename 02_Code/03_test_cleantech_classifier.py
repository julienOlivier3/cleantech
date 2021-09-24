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

import config
import re
import string
import pandas as pd
import numpy as np
import pickle as pkl
from pyprojroot import here
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# # Data preparation 

# Encoding issue here!
df = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))

df.shape

# There are some non-English texts (should not be there as they were translated in 01_access_patstat.py) <- rerun. For now drop these.
df = df.loc[df.ABSTRACT_LANG=='en',:]

# Drop patents with missing entries
df = df.loc[df.ABSTRACT.notnull(),:]

# Drop patents with very short (uninformative) patent abstracts
df = df.loc[df.ABSTRACT.apply(len)>30,:]

# Add label in string format
df['Y02_string'] = df.Y02.map({0: ['non_cleantech'], 1: ['cleantech']})

df.Y02.value_counts()

# There more than 400,000 non-cleantech patents and more than 35,000 cleantech patents. Use these as training data for text classification model.

df

# +
# Train-dev-test split
df_train = df.sample(frac=0.8, random_state=333)
df_test = df.loc[~df.APPLN_ID.isin(df_train.APPLN_ID),:]

X_train = df_train.sample(1000).ABSTRACT.values
X_test = df_test.sample(1000).ABSTRACT.values
y_train = df_train.sample(1000).Y02.values
y_test = df_test.sample(1000).Y02.values
# -

# Load semantic technology spaces

# Read topic-proba-df
df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words.txt'), sep='\t', encoding='utf-8')

# Read semantic vectors from disk
semantic_vectors = pkl.load(open(here(r'.\03_Model\temp\semantic_vectors.pkl'), 'rb'))

# Read word embeddings
embeddings_index = {}
with open(config.PATH_TO_GLOVE + '/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))


# # Functions 

def word_list_to_embedding_array(word_list):
    # Extract word embedding if exist, else return None
    embedding_list = [list(embeddings_index.get(word, [])) for word in word_list]
    # Drop None from resulting list
    embedding_list = list(filter(None, embedding_list))
    # Create numpy array
    embeddings = np.array(embedding_list)
    return(embeddings)


def get_semantic_vectors(technology, n_words):
    return(semantic_vectors[technology][0:n_words,])


# # Testing on hold out patent abstracts 

df_test

df_test.Y02.value_counts(dropna=False)

# Start testing here:

df_test = pd.concat([df_test.loc[df_test.Y02==1].head(5000), df_test.loc[df_test.Y02==0].head(5000)], axis=0)

df_test.reset_index(drop=True, inplace=True)

len(df_test)

stoplist = ['for', 'a', 'of', 'the', 'and', 'to', 'in', 'at', 'an', 'on', 'this', 'is', 'are', 'it', 'the', 'and/or', 'i', 'wt', 'or', 'from', 'first', 'least']

# Drop tokens of form: '(345)'
pattern1 = re.compile("^\(\d{1,}\)$")

df_test

# +
temp = []
for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], position=0, leave=True):
#for index in tqdm(range(df_test.shape[0]), position=0, leave=True):
    #row = df_test.iloc[index]
    clean_document = row.ABSTRACT.split()
    label = row.Y02
    importance = row.Y02_imp

    # First round of text cleaning
    clean_document = [token.lower().rstrip(string.punctuation).lstrip(string.punctuation) for token in clean_document 
                      if not token.isdecimal() and not pattern1.match(token) and not all([j in string.punctuation for j in [c for c in token]]) and len(token)>1]
    # Second round of string cleaning removing stop words
    clean_document = [token for token in clean_document if token not in stoplist]
        
    # Create word embedding matrix
    patent_embedding = word_list_to_embedding_array(clean_document)
    len_patent_embedding = len(patent_embedding)
    
    # Calculate proximity to all clean technology semantic spaces
    for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
        for n_words in range(10, 5000+1, 100):
            technology_embedding = get_semantic_vectors(y02, n_words)
            
            # Calculate cosine similarity between all permutations of patent vector space and technology semantic vector space
            similarity = np.round_(cosine_similarity(patent_embedding, technology_embedding).flatten(), decimals=5)
            similarity_mean = similarity.mean()
        
            # Calculate number of exact word matches
            n_exact = (similarity == 1).sum()
            n_exact_norm = n_exact/len_patent_embedding
        
            temp.append([index, label, y02, importance, n_words, similarity_mean, n_exact, n_exact_norm, n_exact_norm+similarity_mean])
    #if index==5:
    #    break
            
df_temp = pd.DataFrame(temp, columns=['ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'])
