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
import pandas as pd
import numpy as np
import pickle as pkl
from pyprojroot import here
from tqdm import tqdm
from sklearn.metrics import classification_report

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
df_train = df.sample(frac=0.8)
df_test = df.loc[~df.APPLN_ID.isin(df_train.APPLN_ID),:]

X_train = df_train.sample(1000).ABSTRACT.values
X_test = df_test.sample(1000).ABSTRACT.values
y_train = df_train.sample(1000).Y02.values
y_test = df_test.sample(1000).Y02.values

# + [markdown] tags=[] heading_collapsed="true"
# # Transformer language model 
# -

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

# Let us give it a shot and do the classification with a pretrained model without finetuning (zero-shot-classification).

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# +
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(tokens)
tf.math.softmax(output.logits, axis=-1)

# +
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
# -

sample_sequences = list(df_train.loc[df_train.Y02==0,'ABSTRACT'].sample(10).values)
sample_sequences.extend(list(df_train.loc[df_train.Y02==1,'ABSTRACT'].sample(10).values))

# %%time
classifier(
    sample_sequences,
    candidate_labels=["cleantech", "non_cleantech"],
)

# The problem of transformer models is that they are made for shorter sequences of text. Later on in the project the classifier is supposed to be applied on longer texts from corporate websites. For this purpose a more traditional text classification with heuristics from information retrieval (e.g. tf-idf) will be more suitable. 

# + [markdown] tags=[] heading_collapsed="true"
# # Tf-idf text classification model
# -

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

# +
# Vectorize text
vectorizer = TfidfVectorizer(
                encoding = 'utf-8',
                analyzer = 'word',
                #preprocessor=text_standardization
                token_pattern = r'\S+',
                #max_df = 0.975
                #min_df = 0.025
                use_idf = True
                #max_features = 20000
                            )
# Train vectorizer
vectorizer.fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# -

# Train model
model = GradientBoostingClassifier(random_state = 333)
model.fit(X_train_tfidf, y_train)

# +
# Create predictions
y_train_pred = model.predict_proba(X_train_tfidf)
df_train_pred = pd.DataFrame(y_train_pred, columns=['prob_non-Y02', 'prob_Y02'])
df_train_pred['true_Y02'] = y_train
df_train_pred

y_test_pred = model.predict_proba(X_test_tfidf)
df_test_pred = pd.DataFrame(y_test_pred, columns=['prob_non-Y02', 'prob_Y02'])
df_test_pred['true_Y02'] = y_test
df_test_pred
# -

# Distribution preciction probabilities
df_test_pred.prob_Y02.plot(kind='hist', bins=100, logy=True)

# +
# Turn probas into final predictions
threshhold=0.5

df_train_pred['pred_Y02'] = (df_train_pred.prob_Y02 > threshhold).map({False: 0, True: 1})
df_train_pred

df_test_pred['pred_Y02'] = (df_test_pred.prob_Y02 > threshhold).map({False: 0, True: 1})
df_test_pred
# -

print(classification_report(y_true = df_test_pred.true_Y02, y_pred = df_test_pred.pred_Y02, target_names=['non_cleantech', 'cleantech']))

# Decent precision but low recall.

# See [here](https://github.com/julienOlivier3/Text_Classification_Capstone) for alternatively training a neural network.

# + [markdown] tags=[]
# # Labelled Topic Model 
# -

import tomotopy as tp

stoplist = ['for', 'a', 'of', 'the', 'and', 'to', 'in', 'at', 'an', 'on', 'this', 'is', 'are', 'it', 'the', 'and/or', 'i', 'wt', 'or', 'from', 'first', 'least']

# Drop tokens of form: '(345)'
import re
import string
pattern1 = re.compile("^\(\d{1,}\)$")

# +
# ToDo: Add lemmatization here
model = tp.PLDAModel(tw=tp.TermWeight.IDF, topics_per_label=1, 
                     #latent_topics=8,
                     seed=333)

for index, row in tqdm(df_train.iterrows()):
    clean_document = row.ABSTRACT.split()
    labels = row.CPC

    # First round of text cleaning
    clean_document = [token.lower().rstrip(string.punctuation).lstrip(string.punctuation) for token in clean_document 
                      if not token.isdecimal() and not pattern1.match(token) and not all([j in string.punctuation for j in [c for c in token]]) and len(token)>1]
    # Second round of string cleaning removing stop words
    clean_document = [token for token in clean_document if token not in stoplist]
        
    model.add_doc(clean_document, labels=labels)
# -

model.burn_in = 5
print('Starting training model:')
for i in range(0, 100, 10):
    model.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

model.summary(topic_word_top_n=10)


def get_prediction_df(df, topic_dist, t_names):
        
    df_output = df.copy()
    n_rows = len(df)
    n_topics = len(t_names)
    
    y02s = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']
    non_y02s = sorted([t for t in t_names if t not in y02s])
    y02_ids = [t_names.index(y02) for y02 in y02s]
    non_y02s_ids = [t_names.index(l) for l in non_y02s]
    
    # Probability distribution across cleantech patents
    dist_y02 = [{y02s[i]: round(topic_dist[c][y02_ids[i]], 5) for i in range(len(y02s))} for c in range(n_rows)]
    df_output['Y02_PRED_dict'] = dist_y02
    
    # Probability distribution across non-cleantech patents
    #dist_non_y02 = [{non_y02s[i]: round(topic_dist[c][non_y02_ids[i]],3) for i in range(len(non_y02s))} for c in range(n_rows)]
    #df_output['NON_Y02_PRED_dict'] = dist_non_y02
    
    # Cleantech importance
    df_output['Y02_PRED_imp'] = [round(sum(c.values()), 5) for c in dist_y02]

    return df_output


# Create list of topic names in right order
t_names = [t[0] for t in doc_inst[0].labels]
#[t_names.append('L' + str(lt)) for lt in range(1,model.latent_topics+1)]
print(t_names)

# +
# %time
doc_inst = []
for index, row in tqdm(df_test.iterrows()):
    clean_document = row.ABSTRACT.split()
    labels = row.CPC

    # First round of text cleaning
    clean_document = [token.lower().rstrip(string.punctuation).lstrip(string.punctuation) for token in clean_document 
                      if not token.isdecimal() and not pattern1.match(token) and not all([j in string.punctuation for j in [c for c in token]]) and len(token)>1]
    # Second round of string cleaning removing stop words
    clean_document = [token for token in clean_document if token not in stoplist]
        
    doc_inst.append(model.make_doc(clean_document))
    
#doc_inst = [model.make_doc(row.ABSTRACT.split()) for index, row in tqdm(df_train.iterrows())]
topic_dist, ll = model.infer(doc_inst, iter = 100, together = True)
# -

len(topic_dist), ll

df_test_pred = get_prediction_df(df_test, topic_dist, t_names)

df_test_pred.Y02_PRED_imp.plot(kind='hist', bins=100, logy=True)


def classification_res(df, target_names, threshold=0.5):
    y_true = np.array(df.Y02).astype(int)
    y_pred = np.array(df.Y02_PRED_imp >= threshold).astype(int)
    
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


[print(t, classification_res(df_test_pred, target_names=['non_cleantech', 'cleantech'], threshold=t)) for t in np.arange(0.0, 1.0, 0.1)]

# Discriminating between cleantech and non-cleantech patents does not work well with topic models.

df_train1.append(df_train2)

df_train1 = df_train.loc[df_train.Y02==1,:]
df_train2 = df_train.loc[df_train.Y02==0,:].sample(len(df_train1))
df_temp = df_train1.append(df_train2)

# +
model = tp.PLDAModel(tw=tp.TermWeight.IDF, topics_per_label=1, 
                     #latent_topics=8,
                     seed=333)

for index, row in tqdm(df_temp.iterrows()):
    clean_document = row.ABSTRACT.split()
    labels = row.CPC

    # First round of text cleaning
    clean_document = [token.lower().rstrip(string.punctuation).lstrip(string.punctuation) for token in clean_document 
                      if not token.isdecimal() and not pattern1.match(token) and not all([j in string.punctuation for j in [c for c in token]]) and len(token)>1]
    # Second round of string cleaning removing stop words
    clean_document = [token for token in clean_document if token not in stoplist]
        
    model.add_doc(clean_document, labels=labels)
# -

model.burn_in = 5
print('Starting training model:')
for i in range(0, 100, 10):
    model.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

model.summary(topic_word_top_n=10)

# Create list of topic names in right order
t_names = [t[0] for t in doc_inst[0].labels]
#[t_names.append('L' + str(lt)) for lt in range(1,model.latent_topics+1)]
print(t_names)

df_test1 = df_test.loc[df_test.Y02==1,:]
df_test2 = df_test.loc[df_test.Y02==0,:].sample(len(df_test1))
df_temp = df_test1.append(df_test2)

# +
# %time
doc_inst = []
for index, row in tqdm(df_temp.iterrows()):
    clean_document = row.ABSTRACT.split()
    labels = row.CPC

    # First round of text cleaning
    clean_document = [token.lower().rstrip(string.punctuation).lstrip(string.punctuation) for token in clean_document 
                      if not token.isdecimal() and not pattern1.match(token) and not all([j in string.punctuation for j in [c for c in token]]) and len(token)>1]
    # Second round of string cleaning removing stop words
    clean_document = [token for token in clean_document if token not in stoplist]
        
    doc_inst.append(model.make_doc(clean_document))
    
#doc_inst = [model.make_doc(row.ABSTRACT.split()) for index, row in tqdm(df_train.iterrows())]
topic_dist, ll = model.infer(doc_inst, iter = 100, together = True)
# -

len(topic_dist), ll

df_test_pred = get_prediction_df(df_temp, topic_dist, t_names)

df_test_pred.Y02_PRED_imp.plot(kind='hist', bins=100, logy=True)

[print(t, classification_res(df_test_pred, target_names=['non_cleantech', 'cleantech'], threshold=t)) for t in np.arange(0.0, 1.0, 0.1)]

# A balanced training and test data set worsens results further.

# Labelled LDA models rather serve as way to build semantic descriptions of clean technologies. Maybe still a minor research contribution?

# Idea: Take the semantic description of the clean technologies (i.e. the most relevant words per topic in the topic model) as basis to scan the corporate websites. There are two options for this approach:
# - simple scan of the websites for keywords
# - scan of the websites in vector space, i.e. build clean technology vectors based on most relevant words and calculate similarity to the websites vectorised words.

# For this purpose, extract the most relevant words per topic from the topic model.

model.topic_label_dict

n_relevant_words = 10000
df_topic_words = pd.DataFrame(columns = ['Topic', 'Word', 'Prob'])
for ind, topic in enumerate(model.topic_label_dict):
    temp = pd.DataFrame(model.get_topic_words(topic_id=ind, top_n=n_relevant_words), columns=['Word', 'Prob'])
    temp['Topic'] = topic
    df_topic_words = df_topic_words.append(temp)

df_topic_words

# Alternative look at data:

pd.concat([pd.DataFrame(model.get_topic_words(topic_id=ind, top_n=10000), columns=[topic+'_words', topic+'_prob']) for ind, topic in enumerate(model.topic_label_dict)], axis=1)

# Get a semantic representation in vector space.

df_topic_words.loc[df_topic_words.Topic=='Y02C']

embeddings_index = {}
with open(config.PATH_TO_GLOVE + '/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))

semantic_vectors = {}
for topic in model.topic_label_dict:
    temp = []
    for word in tqdm(df_topic_words.loc[df_topic_words.Topic==topic, 'Word']):
        embedding = embeddings_index.get(word, None)
        if isinstance(embedding, np.ndarray):
            temp.append(list(embedding))
        vec = np.array(temp)
    semantic_vectors[topic] = vec

# Calculate cosine similarity:

# cosine similarity = $\frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$

# +
A = np.array([[1, 0.2, 0.1],
             [0.8, 0.1, 0.2],
             [-0.8, -0.1, -0.2]])

B = np.array([[0.3, 0.2, 0.2],
             [0.8, 1, -0.2],
             [-0.1, -0.1, -0.2],
             [0.8, -0.1, -0.8],
             [0.5, -0.5, -0.9]])


# -

def cosine_similarity_vectors(v1, v2):
    numerator=np.dot(v1, v2)
    denumerator1 = np.sqrt(np.sum(np.square(v1)))
    denumerator2 = np.sqrt(np.sum(np.square(v2)))
    return(numerator*1/(denumerator1*denumerator2))


cosine_similarity_vectors(A[0], A[1])


def cosine_similarity_matrix(M):
    similarity = np.dot(M, M.T)
    
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return(cosine.T * inv_mag)


cosine_similarity_matrix(A)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(A)

cosine_similarity(A, B)

cosine_similarity_vectors(A[0], B[3])

cosine_similarity(A, B).mean()

temp=[]
for ra in range(A.shape[0]):
    for rb in range(B.shape[0]):
        temp.append(cosine_similarity_vectors(A[ra], B[rb]))
np.mean(temp)


# Mean of upper/lower triangle of resulting matrix gives overall similarity between arrays in ndarray.

def cosine_to_mean(M_cos):
    return(M_cos[np.tril_indices(M_cos.shape[0], k=-1)])


cosine_to_mean(cosine_similarity_matrix(A))

cosine_to_mean(cosine_similarity_matrix(A)).mean()

# Now apply to most relevant topic words yielded from topic model.

y02c_cos = cosine_to_mean(cosine_similarity(y02c))

y02c_cos.mean()
