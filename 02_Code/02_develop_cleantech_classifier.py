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
import seaborn as sns

df = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))

df.shape

df.head(3)

# Drop patents with very short (uninformative) patent abstracts
df = df.loc[df.LEMMAS.apply(len)>=1]

df.shape

# Add label in string format
df['Y02_string'] = df.Y02.map({0: ['non_cleantech'], 1: ['cleantech']})

df.Y02.value_counts()

# There more than 500,000 non-cleantech patents and more than 40,000 cleantech patents. Use these as training data for text classification model.

# Lemmatization of data -> datum is wrong.
# Moreover, lemmas should be lowercased and should consist of more than one character
tqdm.pandas()
df['LEMMAS'] = df.LEMMAS.progress_apply(lambda x: [lemma.lower().replace('datum', 'data') for lemma in x if len(lemma)>1])

# Count number of lemmas
vocs = set()
for index, row in tqdm(df.iterrows()):
    voc = set(row.LEMMAS)
    vocs.update(voc)
#    if index > 100:
#        break
len(vocs)

# The corpus comprises almost 350,000 distinct lemmas.

# Count number of patents in different Y02 classes
y02_count = {}
for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    y02_count[y02] = df.CPC.apply(lambda x: y02 in x).sum()

y02_count

# Number of Y02 patents in some Y02 classes low (Y02C, Y02D). Make sure that the test set contains a minimum number samples for each Y02 class.

# +
# Train-dev-test split
df_train = df.sample(frac=0.8, random_state=333)
df_test = df.loc[~df.APPLN_ID.isin(df_train.APPLN_ID),:]

X_train = df_train.sample(1000).ABSTRACT.values
X_test = df_test.sample(1000).ABSTRACT.values
y_train = df_train.sample(1000).Y02.values
y_test = df_test.sample(1000).Y02.values
# -

df_test.shape

# Save test data to disk for future reference
with open(here(r'.\03_Model\temp\df_test.pkl'), 'wb') as f:
    pkl.dump(df_test, f)

# + [markdown] tags=[] heading_collapsed="true" jp-MarkdownHeadingCollapsed=true
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

# + [markdown] tags=[] heading_collapsed="true"
# # Labelled Topic Model 
# -

import tomotopy as tp

stoplist = ['and/or', '/h', 't1', 'dc', 'mm', 'wt', '113a', '115a', 'ofdm', 'lpwa']

# Instantiate labelled LDA model
model = tp.PLDAModel(tw=tp.TermWeight.IDF, topics_per_label=1, 
                     #latent_topics=8,
                     seed=333)

# Add documents to model
for index, row in tqdm(df_train.iterrows()):
    clean_document = row.LEMMAS
    # Remove some additional stopwords
    clean_document = [token for token in clean_document if token not in stoplist]
    # Remove Y04 and Y10 tag
    labels = [cpc for cpc in row.CPC if cpc not in ['Y04', 'Y10']]
    # Add document and labels to model
    model.add_doc(clean_document, labels=labels)

model.burn_in = 5
print('Start training model:')
for i in range(0, 100, 10):
    model.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

model.summary(topic_word_top_n=10)

# + active=""
# # Save model to disk (deprecated)
# model.save(here(r'.\03_Model\temp\PLDA_model.bin').__str__(), full=True)
# -

# Labelled LDA models rather serve as way to build semantic descriptions of clean technologies (generative not discriminative model). Maybe still a minor research contribution?

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

# + active=""
# # Save to disk
# df_topic_words.to_csv(here(r'.\03_Model\temp\df_topic_words.txt'), sep='\t', encoding='utf-8')
# -

# Read topic-proba-df
df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words.txt'), sep='\t', encoding='utf-8')

# Get a semantic representation in vector space.

df_topic_words.loc[df_topic_words.Topic=='Y02C']

# %%time
embeddings_index = {}
with open(config.PATH_TO_GLOVE + '/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))

# Create set of technology words for which pre-trained word embeddings exist
technology_words = set(df_topic_words.Word.values)
embedded_words = set(embeddings_index.keys())
embedded_technology_words = embedded_words.intersection(technology_words)

len(technology_words), len(embedded_words), len(embedded_technology_words)

# Reduce technology words to those with existing word embeddings
df_topic_words = df_topic_words.loc[df_topic_words.Word.isin(embedded_technology_words)]

df_topic_words.shape

# Create dataframe containing only words for which embeddings exist for all technology classes
df_excel = pd.DataFrame()
for topic in model.topic_label_dict:
    df_temp = df_topic_words.loc[df_topic_words.Topic==topic][:5000]
    df_temp = df_temp.drop(columns=['Topic']).rename(columns={'Word': topic+'_word', 'Prob': topic+'_prob'}).reset_index(drop=True)
    print(topic, '\t', df_temp.shape)
    df_excel = pd.concat([df_excel, df_temp], axis=1)

df_excel

# + active=""
# # Save to disk
# df_excel.to_excel(here(r'.\03_Model\temp\df_topic_words.xlsx'), encoding='utf-8')
# -

semantic_vectors = {}
for topic in model.topic_label_dict:
    temp = []
    for word in tqdm(df_topic_words.loc[df_topic_words.Topic==topic, 'Word']):
        embedding = embeddings_index.get(word, None)
        if isinstance(embedding, np.ndarray):
            temp.append(list(embedding))
        vec = np.array(temp)
    semantic_vectors[topic] = vec

# Number of top words with existing word vectors in Glove.

for topic in model.topic_label_dict:
    print(topic, '\t' , len(semantic_vectors[topic]))

# + active=""
# # Save semantic vectors to disk
# with open(here(r'.\03_Model\temp\semantic_vectors.pkl'), 'wb') as f:
#     pkl.dump(semantic_vectors, f)
# -

# Read semantic vectors from disk
semantic_vectors = pkl.load(open(here(r'.\03_Model\temp\semantic_vectors.pkl'), 'rb'))

# From this picture it makes possibly sense to restrict the technology clusters in semantic vector space to the 4,000 most relevant terms.

semantic_vectors4000 = {}
for topic in list(semantic_vectors.keys()):
    vec = semantic_vectors[topic][0:4000,]
    semantic_vectors4000[topic] = vec

for topic in list(semantic_vectors.keys()):
    print(topic, '\t' , len(semantic_vectors4000[topic]))


# Write a general function to extract semantic vectors of different size:

def get_semantic_vectors(topic, n_words):
    return(semantic_vectors[topic][0:n_words,])


get_semantic_vectors('Y02C', 3)

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

# Let us assume the semantic space of "carbon capture and storage technologies" comprises the words $v_{CSS} = [carbon, dioxide, absorb, ...]$.
# A company description of a firm that applies these technologies may comprise the words $v_{firm} = [co2, reduce, ...]$.
#
# In essence, a company descriptions is likely to be less technical than a patent description. Still, the technological space and the company description point to the same construct: the extraction of CO2 from space. So it desirable to have a high similarity between $v_{CSS}$ and $v_{firm}$.

v_CSS = [embeddings_index[word] for word in ['carbon', 'dioxide', 'absorb']]
v_firm = [embeddings_index[word] for word in ['co2', 'reduce']]

cos1 = cosine_similarity_vectors(embeddings_index['carbon'], embeddings_index['co2'])
cos2 = cosine_similarity_vectors(embeddings_index['carbon'], embeddings_index['reduce'])
cos3 = cosine_similarity_vectors(embeddings_index['dioxide'], embeddings_index['co2'])
cos4 = cosine_similarity_vectors(embeddings_index['dioxide'], embeddings_index['reduce'])
cos5 = cosine_similarity_vectors(embeddings_index['absorb'], embeddings_index['co2'])
cos6 = cosine_similarity_vectors(embeddings_index['absorb'], embeddings_index['reduce'])

cos1, cos2, cos3, cos4, cos5, cos6

np.mean([cos1, cos2, cos3, cos4, cos5, cos6])

cosine_similarity(v_CSS, v_firm).mean()

# Note that cosine similarity between word embeddings has value range [-1, 1]. Values < 0 indicate dissimilarity and should thus be set to 0 in a similarity measure following the approach in this [paper](https://aclanthology.org/N19-1181.pdf).

temp = cosine_similarity(v_CSS, v_firm).flatten()
temp[temp>0.7]=1
temp

# Alternatively, one can also take the mean of both vectors first and caclculate cosine similarity then.

cosine_similarity_vectors(np.array(v_CSS).mean(axis=0), np.array(v_firm).mean(axis=0))

from gensim import matutils
np.dot(matutils.unitvec(np.array(v_CSS).mean(axis=0)), matutils.unitvec(np.array(v_firm).mean(axis=0)))


# It looks like a decent similarity between the two vectors exists.

# Mean of upper/lower triangle of resulting matrix gives overall similarity between arrays in ndarray.

def cosine_to_mean(M_cos):
    return(M_cos[np.tril_indices(M_cos.shape[0], k=-1)])


cosine_to_mean(cosine_similarity_matrix(A))

cosine_to_mean(cosine_similarity_matrix(A)).mean()

# Now apply to most relevant topic words yielded from topic model.

df_topic_words.loc[df_topic_words.Topic=='Y02C'].head(10)

for word in df_topic_words.loc[df_topic_words.Topic=='Y02C'].head(10).Word:
    print('gas +', word, '\t', cosine_similarity_vectors(embeddings_index['gas'], embeddings_index[word]))

# Cosine similarity between word vectors within Y02E class
cosine_to_mean(cosine_similarity(semantic_vectors5000['Y02E'])).mean(), np.median(cosine_to_mean(cosine_similarity(semantic_vectors5000['Y02E'])))

plt.hist(cosine_to_mean(cosine_similarity(semantic_vectors5000['Y02E'])), bins='auto')
plt.show()

# Cosine similarity between word vectors Y02E and Y02C classes
cosine_similarity(semantic_vectors5000['Y02E'], semantic_vectors5000['Y02C']).mean(), np.median(cosine_similarity(semantic_vectors5000['Y02E'], semantic_vectors5000['Y02C']))

plt.hist(cosine_similarity(semantic_vectors5000['Y02E'], semantic_vectors5000['Y02C']).flatten(), bins = 'auto') 
plt.show()

# Cosine similarity between word vectors Y02E and A classes
cosine_similarity(semantic_vectors5000['Y02E'], semantic_vectors5000['C']).mean(), np.median(cosine_similarity(semantic_vectors5000['Y02E'], semantic_vectors5000['C']))

plt.hist(cosine_similarity(semantic_vectors5000['Y02E'], semantic_vectors5000['C']).flatten(), bins = 'auto') 
plt.show()

# Look at similarity given a sentence.

sent = 'We have 10 years of experience with modular capture plants, and over 50,000 operating hours, capturing CO2 from WtE, gas and coal fired power plants, refineries and cement industries. Our Advanced Carbon Capture technology has been applied across a wide range of industries onshore. We also offer an offshore version addressing emissions from oil and gas production. Our products and solutions cover both mid-range and large-scale emitters.'
sent_embedding = np.array([list(embeddings_index[word]) for word in re.sub(r'\.|\,', '', sent).lower().split()])
cosine_similarity(semantic_vectors5000['Y02C'], sent_embedding).flatten().mean()

plt.hist(cosine_similarity(semantic_vectors5000['Y02C'], sent_embedding).flatten(), bins = 'auto') 
plt.show()

# How does within and between similarity depend on the number of words in the technologies' semantic space?

# Work with semantic vectors of different sizes (limit the number of top words) and return mean and median cosine similarity.
temp = []
for topic in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    for n_words in tqdm(range(10, 5000+1, 10)):
        Y02_embedding = get_semantic_vectors(topic, n_words)
        similarity = cosine_similarity(Y02_embedding, sent_embedding).flatten()
        temp.append([topic, n_words, similarity.mean(), np.median(similarity)])
df_temp = pd.DataFrame(temp, columns=['y02', 'n_words', 'mean', 'median'])

sns.lineplot(data=df_temp, x='n_words', y='median', hue='y02')

# One way to reduce noise is to ignore cosine similarities of word pairs below a certain threshold or to incorporate exact matches in the similarity metric.

temp1 = get_semantic_vectors('Y02C', 5)
temp2 = sent_embedding
similarity = cosine_similarity(temp1, temp2).flatten()
similarity[np.argsort(-similarity)[:100]]
(np.round_(similarity[np.argsort(-similarity)[:100]], decimals=3) == 1).sum()
len(sent_embedding)

# Work with semantic vectors of different sizes (limit the number of top words) and return mean and median cosine similarity.
temp = []
len_sent = len(sent_embedding)
for topic in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    for n_words in tqdm(range(10, 5000+1, 10)):
        # Get semantic technology descriptions in vector space given number of top words
        Y02_embedding = get_semantic_vectors(topic, n_words)
        
        # Calculate cosine similarity between all permutations of technology semantic vector space and sentence vector space
        similarity = np.round_(cosine_similarity(Y02_embedding, sent_embedding).flatten(), decimals=3)
        
        # Calculate number of exact word matches
        n_exact = (similarity == 1).sum()
        n_exact_norm = n_exact/len_sent
        
        temp.append([topic, n_words, similarity.mean(), np.median(similarity),
                     np.quantile(similarity, q=0.9), np.quantile(similarity, q=0.99), np.quantile(similarity, q=0.999),
                     n_exact, n_exact_norm, n_exact_norm+similarity.mean()])
df_temp = pd.DataFrame(temp, columns=['y02', 'n_words', 'mean', 'median',
                                     'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'])

df = pd.melt(df_temp, id_vars=['y02', 'n_words'], value_vars=['mean', 'median', 'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'], var_name='measure', value_name='value')

df.head(3)

sns.relplot(
    data=df,
    x="n_words", y="value",
    hue="y02",  col="measure", kind="line",
    col_wrap=3, facet_kws=dict(sharey=False)
)

# Exact matches of words
list_sent = re.sub(r'\.|\,', '', sent).lower().split()
[word for word in list_sent if word in df_topic_words.loc[df_topic_words.Topic=='Y02C'].Word[:5000].values]

# How does it look like if we only take top words from one topic?

# Work with semantic vectors of different sizes (limit the number of top words) and return mean and median cosine similarity.
top_words = ['gas', 'absorption', 'dioxide', 'carbon', 'co2']
sent_embedding = np.array([list(embeddings_index[word]) for word in top_words])
temp = []
len_sent = len(sent_embedding)
for topic in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    for n_words in tqdm(range(10, 5000+1, 10)):
        # Get semantic technology descriptions in vector space given number of top words
        Y02_embedding = get_semantic_vectors(topic, n_words)
        
        # Calculate cosine similarity between all permutations of technology semantic vector space and sentence vector space
        similarity = np.round_(cosine_similarity(Y02_embedding, sent_embedding).flatten(), decimals=3)
        
        # Calculate number of exact word matches
        n_exact = (similarity == 1).sum()
        n_exact_norm = n_exact/len_sent
        
        temp.append([topic, n_words, similarity.mean(), np.median(similarity),
                     np.quantile(similarity, q=0.9), np.quantile(similarity, q=0.99), np.quantile(similarity, q=0.999),
                     n_exact, n_exact_norm, n_exact_norm+similarity.mean()])
df_temp = pd.DataFrame(temp, columns=['y02', 'n_words', 'mean', 'median',
                                     'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'])

df = pd.melt(df_temp, id_vars=['y02', 'n_words'], value_vars=['mean', 'median', 'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'], var_name='measure', value_name='value')

sns.relplot(
    data=df,
    x="n_words", y="value",
    hue="y02",  col="measure", kind="line",
    col_wrap=3, facet_kws=dict(sharey=False)
)

# Even if one takes the same words as listed in the semantic technology space, there is proximity but not extremely clear proximity. Why? Because the words within the semantic technology space have in part a great dissimilarity based on the pre-trained word vectors. It may be worth here to train own technology specific word vectors.

# What if one takes a completely non-technical business model description?

sent = 'Our little café with a pastry shop is not called Patisserie because it is exclusively French. On the contrary, we offer you a mixture of more than just one nation. We have made it our specialty to offer a small mix of French, English and German confectionery'
sent_embedding = np.array([list(embeddings_index[word]) for word in re.sub(r'\.|\,', '', sent).lower().split()])
cosine_similarity(semantic_vectors5000['Y02C'], sent_embedding).flatten().mean()

# Work with semantic vectors of different sizes (limit the number of top words) and return mean and median cosine similarity.
temp = []
len_sent = len(sent_embedding)
for topic in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    for n_words in tqdm(range(10, 5000+1, 10)):
        # Get semantic technology descriptions in vector space given number of top words
        Y02_embedding = get_semantic_vectors(topic, n_words)
        
        # Calculate cosine similarity between all permutations of technology semantic vector space and sentence vector space
        similarity = np.round_(cosine_similarity(Y02_embedding, sent_embedding).flatten(), decimals=3)
        
        # Calculate number of exact word matches
        n_exact = (similarity == 1).sum()
        n_exact_norm = n_exact/len_sent
        
        temp.append([topic, n_words, similarity.mean(), np.median(similarity),
                     np.quantile(similarity, q=0.9), np.quantile(similarity, q=0.99), np.quantile(similarity, q=0.999),
                     n_exact, n_exact_norm, n_exact_norm+similarity.mean()])
df_temp = pd.DataFrame(temp, columns=['y02', 'n_words', 'mean', 'median',
                                     'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'])

df = pd.melt(df_temp, id_vars=['y02', 'n_words'], value_vars=['mean', 'median', 'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'], var_name='measure', value_name='value')

# + tags=[]
sns.relplot(
    data=df,
    x="n_words", y="value",
    hue="y02",  col="measure", kind="line",
    col_wrap=3, facet_kws=dict(sharey=False)
)
# -

# Which words do match with semantic technology spaces?

# Exact matches of words
list_sent = re.sub(r'\.|\,', '', sent).lower().split()
matched_words = [word for word in list_sent if word in df_topic_words.loc[df_topic_words.Topic=='Y02C'].Word.values]
matched_words

# Exact matches are only words which do not describe the semantic space of the respective technology at all (maybe except of the word mixture). This shows that semantic spaces of technologies need to be cleaned manually to a certain extent. How does proximity measure looks like if these words were not part of the semantic technology space?

sent_embedding.shape

sent_list=[word for word in list_sent if word not in matched_words]

sent_embedding = np.array([list(embeddings_index[word]) for word in list_sent if word not in matched_words])

sent_embedding.shape

# Work with semantic vectors of different sizes (limit the number of top words) and return mean and median cosine similarity.
temp = []
len_sent = len(sent_embedding)
for topic in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    for n_words in tqdm(range(10, 5000+1, 10)):
        # Get semantic technology descriptions in vector space given number of top words
        Y02_embedding = get_semantic_vectors(topic, n_words)
        
        # Calculate cosine similarity between all permutations of technology semantic vector space and sentence vector space
        similarity = np.round_(cosine_similarity(Y02_embedding, sent_embedding).flatten(), decimals=5)
        
        # Calculate number of exact word matches
        n_exact = (similarity == 1).sum()
        n_exact_norm = n_exact/len_sent
        
        temp.append([topic, n_words, similarity.mean(), np.median(similarity),
                     np.quantile(similarity, q=0.9), np.quantile(similarity, q=0.99), np.quantile(similarity, q=0.999),
                     n_exact, n_exact_norm, n_exact_norm+similarity.mean()])
df_temp = pd.DataFrame(temp, columns=['y02', 'n_words', 'mean', 'median',
                                     'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'])

df = pd.melt(df_temp, id_vars=['y02', 'n_words'], value_vars=['mean', 'median', 'q9', 'q99', 'q999', 'n_exact', 'n_exact_norm', 'n_exact_norm_mean'], var_name='measure', value_name='value')

sns.relplot(
    data=df,
    x="n_words", y="value",
    hue="y02",  col="measure", kind="line",
    col_wrap=3, facet_kws=dict(sharey=False)
)

# If no search terms match exactly, then the proximity measure n_exact_norm+similarity.mean() converge towards the mean cosine similarity across all permutations. Proximity above mean could thus be a filter for considering a text as technology related.
