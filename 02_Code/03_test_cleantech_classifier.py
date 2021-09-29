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
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# + [markdown] heading_collapsed="true" tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Functions 
# -

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


# + [markdown] heading_collapsed="true" tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Data preparation 
# -

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

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Testing on hold out patent abstracts 

# + [markdown] tags=[] heading_collapsed="true"
# ## Calculate technological proximity 
# -

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
    ind = row.APPLN_ID

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
        
            temp.append([ind, label, y02, importance, n_words, similarity_mean, n_exact, n_exact_norm, n_exact_norm+similarity_mean])
    #if index==5:
    #    break
            
df_temp = pd.DataFrame(temp, columns=['ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'])
# -

df_temp

# + active=""
# # Save to disk
# df_temp.to_csv(here(r'.\03_Model\temp\df_validation_patents.txt'), sep='\t', encoding='utf-8')

# + [markdown] tags=[]
# ## Some visualizations 
# -

# Load testing file
df_test_results = pd.read_csv(here(r'.\03_Model\temp\df_validation_patents.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')

df_test_results.head(3)

df_test = pd.melt(df_test_results, id_vars=['LABEL', 'N_WORDS', ], value_vars=['MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'], var_name='measure', value_name='value')

df_test

sns.catplot(
    data=df_test,
    x="N_WORDS", y="value",
    hue="LABEL",  col="measure", kind="box",
    col_wrap=1, sharey=False, sharex=True, height=8, aspect=2
)

# It appears that all proximity measures allow a differentiation between cleantech and non-cleantech patents (with the last one = sum over fraction of exact matches and mean cosine similarity between semantic vector spaces showing this differentiation best.) Nonetheless differentiation could be better. Ideas for improvement clearly exist:
# - training own technology-related word embeddings
# - lemmatization before generating the the semantic vector spaces for the different technology classes
# In any case: The test data can also be used as validation set in order to determine the "best" number of words to define the semantic technology space.
#
# Note: At this point it is no real test set. Let's how the above plots look like on a real hold-out test set.

# Add Y02_dict to test data
df_test = df_test_results.merge(df[['APPLN_ID', 'Y02_dict']], how='left', left_on='APPLN_ID', right_on='APPLN_ID')

df_test['Y02_true'] = df_test.Y02_dict.apply(lambda x: list(x.keys()))

# Analyze how proximity differs compared to non-cleantech patents.

# +
fig, axes = plt.subplots(figsize=(15,50), ncols=1, nrows=8)

for y02, ax in zip(['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W'], axes.flat):
    df_y02 = df_test.loc[df_test.Y02_true.apply(lambda x: y02 in x) & (df_test.Y02 == y02)]
    df_nony02 = df_test.loc[(df_test.LABEL == 0) & (df_test.Y02 == y02)]
    df_temp = pd.concat([df_y02, df_nony02])
    #df_temp = pd.melt(df_temp, id_vars=['LABEL', 'N_WORDS', ], value_vars=['MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'], var_name='measure', value_name='value')
    ax = sns.boxplot(x="N_WORDS", y="N_EXACT_NORM_MEAN", hue="LABEL",
                 data=df_temp, linewidth=1, ax=ax)
    ax.set_title(y02 + ' (based on ' + str(len(df_y02['APPLN_ID'].drop_duplicates())) + ' distinct Y02 patents and 5000 non-Y02 patents)')
plt.show()
# -

# Analyze how proximity differs compared to cleantech patents from other Y02 classes.

fig, axes = plt.subplots(figsize=(15,50), ncols=1, nrows=8)
y02s = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']
for y02, ax in zip(y02s, axes.flat):
    df_y02 = df_test.loc[df_test.Y02_true.apply(lambda x: y02 in x) & (df_test.Y02 == y02)].copy()
    df_y02['LABEL'] = y02
    df_othery02 = df_test.loc[df_test.Y02_true.apply(lambda x: (y02 not in x) & (len(x)>0)) & (df_test.Y02 == y02)].copy()
    df_othery02['LABEL'] = 'other cleantech'
    df_temp = pd.concat([df_y02, df_othery02])
    #df_temp = pd.melt(df_temp, id_vars=['LABEL', 'N_WORDS', ], value_vars=['MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'], var_name='measure', value_name='value')
    ax = sns.boxplot(x="N_WORDS", y="N_EXACT_NORM_MEAN", hue="LABEL",
                 data=df_temp, linewidth=1, ax=ax)
    ax.set_title(y02 + ' (based on ' + str(len(df_y02['APPLN_ID'].drop_duplicates())) + ' distinct ' + y02 + ' patents and ' + str(len(df_othery02['APPLN_ID'].drop_duplicates())) + ' other Y02 patents)')
plt.show()

# Results look good.

# + [markdown] tags=[]
# # Testing on corporate websites 
# -

import util
import config
import pandas as pd
import numpy as np
import re
from pyprojroot import here
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# Function which joins list of strings to one joint string while treating missing values consistently
def create_joint_string(x, columns = ['SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW']):
    return(' '.join([i for i in list(x[columns].values) if not pd.isnull(i)]))    


# +
# Read test data of cleantech and nasdaq firms
df_cleantech = pd.read_csv(here('01_Data/02_Firms/df_cleantech_firms.txt'), sep='\t', encoding='utf-8')
df_cleantech['DESCRIPTION'] = df_cleantech.apply(lambda x: create_joint_string(x), axis=1)
df_cleantech = df_cleantech[['NAME', 'DESCRIPTION']]
df_cleantech['LABEL'] = 'cleantech'

df_nasdaq = pd.read_csv(here('01_Data/02_Firms/df_nasdaq_firms.txt'), sep='\t', encoding='utf-8')
df_nasdaq = df_nasdaq[df_nasdaq.BUSINESS_SUMMARY.notnull()]
df_nasdaq = df_nasdaq[['NAME', 'BUSINESS_SUMMARY']].rename(columns={'BUSINESS_SUMMARY': 'DESCRIPTION'})
df_nasdaq['LABEL'] = 'nasdaq'

# Combine both firm samples in one df
df = pd.concat([df_cleantech, df_nasdaq]).reset_index(drop=True)

# Calculate character length of company descriptions
df['LEN'] = df.DESCRIPTION.apply(len)

df.reset_index(drop=True, inplace=True)
# -

# Analyze length of company descriptions between cleantech and NASDAQ firms.

sns.histplot(data=df, x="LEN", hue="LABEL", bins=30)

# Careful: texts are not equally distributed!

# Conduct lemmatization
df['LEMMA'] = df.DESCRIPTION.apply(lambda x: [lemma.lower() for lemma in string_to_lemma(x)])

df.head(3)

# +
temp = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
#for index in tqdm(range(df_test.shape[0]), position=0, leave=True):
    #row = df_test.iloc[index]
    clean_document = row.LEMMA
    label = row.LABEL
    ind = row.NAME
        
    # Create word embedding matrix
    patent_embedding = word_list_to_embedding_array(clean_document)
    len_patent_embedding = len(patent_embedding)
    
    # Calculate proximity to all clean technology semantic spaces
    for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
        for n_words in [10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]:
            technology_embedding = get_semantic_vectors(y02, n_words)
            
            # Calculate cosine similarity between all permutations of patent vector space and technology semantic vector space
            similarity = np.round_(cosine_similarity(patent_embedding, technology_embedding).flatten(), decimals=5)
            similarity_mean = similarity.mean()
        
            # Calculate number of exact word matches
            n_exact = (similarity == 1).sum()
            n_exact_norm = n_exact/len_patent_embedding
        
            temp.append([ind, label, y02, n_words, similarity_mean, n_exact, n_exact_norm, n_exact_norm+similarity_mean])
    #if index==5:
    #    break
            
df_test_results = pd.DataFrame(temp, columns=['ID', 'LABEL', 'Y02', 'N_WORDS', 'MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'])
# -

df_test = pd.melt(df_test_results, id_vars=['LABEL', 'N_WORDS', ], value_vars=['MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'], var_name='measure', value_name='value')

sns.catplot(
    data=df_test,
    x="N_WORDS", y="value",
    hue="LABEL",  col="measure", kind="box",
    col_wrap=1, sharey=False, sharex=True, height=6, aspect=2
)

# Discrimination between company descriptions looks promising, too!

df_test_results.head(3)

# +
fig, axes = plt.subplots(figsize=(15,50), ncols=1, nrows=8)

for y02, ax in zip(['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W'], axes.flat):
    df_temp = df_test_results.loc[df_test_results.Y02 == y02]
    ax = sns.boxplot(x="N_WORDS", y="N_EXACT_NORM_MEAN", hue="LABEL",
                 data=df_temp, linewidth=1, ax=ax)
    ax.set_title(y02)
plt.show()
