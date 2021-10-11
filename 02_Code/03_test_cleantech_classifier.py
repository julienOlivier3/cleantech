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
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# + [markdown] heading_collapsed="true" tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Functions & Data
# -

# Read topic-proba-df
df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')

# Read semantic vectors from disk
semantic_vectors = pkl.load(open(here(r'.\03_Model\temp\semantic_vectors.pkl'), 'rb'))      # based on word embeddings
semantic_vectors_wavg = pkl.load(open(here(r'.\03_Model\temp\semantic_vectors_wavg.pkl'), 'rb')) # based on averaged word embeddings with word probas as weights
semantic_vectors_bert = pkl.load(open(here(r'.\03_Model\temp\semantic_vectors_bert.pkl'), 'rb')) # based on transformer embeddings

# Read word embeddings
embeddings_index = {}
with open(config.PATH_TO_GLOVE + '/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))


# Calculate cosine similarity between two vectors
def cosine_similarity_vectors(v1, v2):
    numerator=np.dot(v1, v2)
    denumerator1 = np.sqrt(np.sum(np.square(v1)))
    denumerator2 = np.sqrt(np.sum(np.square(v2)))
    return(numerator*1/(denumerator1*denumerator2))


# Function that translates a list of words into a numpy array of word embeddings
# Emebdding_type specifies the embedding strategy: 'we' = nd.array of word embeddings, 'we_avg': weighted average over nd.array of word_embeddings
# 'bert': transformer word embedding
def word_list_to_embedding_array(word_list, embedding_type='we'):
    if embedding_type=='we':
        # Extract word embedding if exist, else return None
        embedding_list = [list(embeddings_index.get(word, [])) for word in word_list]
        # Drop None from resulting list
        embedding_list = list(filter(None, embedding_list))
        # Create numpy array
        embedding = np.array(embedding_list)
    if embedding_type=='we_avg':
        # Extract word embedding if exist, else return None
        embedding_list = [list(embeddings_index.get(word, [])) for word in word_list]
        # Drop None from resulting list
        embedding_list = list(filter(None, embedding_list))
        # Create numpy array
        embedding = np.array(embedding_list).mean(axis=0)
    if embedding_type=='bert':
        # Concatenate list of words to whitespace seperated string
        word_concatenation = ' '.join(str(i) for i in word_list)
        # Create numpy array of sentence embedding
        embedding = model.encode(word_concatenation)        
    return(embedding)


# Function that returns the semantic vector space of a technology with n_words determining the desired size of the semantic vector space
def get_semantic_vectors(technology, n_words, embedding_type='we'):
    if embedding_type=='we':
        return(semantic_vectors[technology][0:n_words,])
    if embedding_type=='bert':
        return(semantic_vectors_bert[technology][n_words])
    if embedding_type=='we_avg':
        return(semantic_vectors_wavg[technology][n_words])


df_test = pd.read_pickle(here(r'.\03_Model\temp\df_test.pkl'))

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Testing on hold out patent abstracts 

# + [markdown] tags=[]
# ## Calculate technological proximity 
# -

df_test.head(3)

# Number of cleantech and non-cleantech patents
df_test.Y02.value_counts(dropna=False)

# Count number of patents in different Y02 classes
y02_count = {}
for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
    y02_count[y02] = df_test.CPC.apply(lambda x: y02 in x).sum()
y02_count

# Start testing here:

df_test.loc[df_test.Y02==1].shape[0]

# Reduce test set so cleantech and non-cleantech patents are balanced
df_cleantech = df_test.loc[df_test.Y02==1]
df_non_cleantech = df_test.loc[df_test.Y02==0]
df_test = pd.concat([df_cleantech, df_non_cleantech.sample(df_cleantech.shape[0], random_state=333)], axis=0)

# Reset index
df_test.reset_index(drop=True, inplace=True)

# Number of patents in test set
len(df_test)

df_test.head(3)


def proximity_testing(df_test, embedding_type):
    stoplist = ['and/or', '/h', 't1', 'dc', 'mm', 'wt', '113a', '115a', 'ofdm', 'lpwa']
    
    temp = []
    
    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], position=0, leave=True):
    #for index in tqdm(range(df_test.shape[0]), position=0, leave=True):
        #row = df_test.iloc[index]
        clean_document = row.ABSTRACT.split()
        # Remove some additional stopwords
        clean_document = [token for token in clean_document if token not in stoplist]
        label = row.Y02
        importance = row.Y02_imp
        ind = row.APPLN_ID

        clean_document = row.LEMMAS
        # Remove some additional stopwords
        clean_document = [token for token in clean_document if token not in stoplist]
        # Remove Y04 and Y10 tag
        labels = [cpc for cpc in row.CPC if cpc not in ['Y04', 'Y10']]

        # Create word embedding matrix
        patent_embedding = word_list_to_embedding_array(clean_document, embedding_type=embedding_type)
        len_patent_embedding = len(patent_embedding)

        # Calculate proximity to all clean technology semantic spaces
        for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
            for n_words in [10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000, 3000, 4000]:
                
                if embedding_type == 'we':
                    technology_embedding = get_semantic_vectors(y02, n_words, embedding_type=embedding_type)

                    # Calculate cosine similarity between all permutations of patent vector space and technology semantic vector space
                    similarity = np.round_(cosine_similarity(patent_embedding, technology_embedding).flatten(), decimals=5)
                    similarity[similarity < 0] = 0
                    similarity_mean = similarity.mean()
                    # Calculate mean embedding and then cosine similarity between both document embeddings
                    similarity_mean2 = cosine_similarity_vectors(patent_embedding.mean(axis=0), technology_embedding.mean(axis=0))
                    similarity_mean2[similarity_mean2 < 0] = 0

                    # Calculate number of exact word matches
                    n_exact = (similarity == 1).sum()
                    n_exact_norm = n_exact/len_patent_embedding

                    temp.append([ind, label, y02, importance, n_words, similarity_mean, similarity_mean2, n_exact, n_exact_norm, n_exact_norm+similarity_mean, n_exact_norm+similarity_mean2])
                    df_prox = pd.DataFrame(temp, columns=['APPLN_ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN', 'MEAN2', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN', 'N_EXACT_NORM_MEAN2'])
                    
                
                if embedding_type == 'we_avg':
                    technology_embedding = get_semantic_vectors(y02, n_words, embedding_type=embedding_type)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity_vectors(patent_embedding, technology_embedding)
                    if similarity < 0:
                        similarity = 0
                    
                    temp.append([ind, label, y02, importance, n_words, similarity])
                    df_prox = pd.DataFrame(temp, columns=['APPLN_ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN_WAVG'])

                
                if embedding_type == 'bert':
                    technology_embedding = get_semantic_vectors(y02, n_words, embedding_type=embedding_type)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity_vectors(patent_embedding, technology_embedding)
                    if similarity < 0:
                        similarity = 0
                    
                    temp.append([ind, label, y02, importance, n_words, similarity])
                    df_prox = pd.DataFrame(temp, columns=['APPLN_ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN_BERT'])
    
    return(df_prox)


df_wavg = proximity_testing(df_test, embedding_type='we_avg')
df_bert = proximity_testing(df_test, embedding_type='bert')

df_wavg

# +
stoplist = ['and/or', '/h', 't1', 'dc', 'mm', 'wt', '113a', '115a', 'ofdm', 'lpwa']
temp = []
for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], position=0, leave=True):
#for index in tqdm(range(df_test.shape[0]), position=0, leave=True):
    #row = df_test.iloc[index]
    clean_document = row.ABSTRACT.split()
    # Remove some additional stopwords
    clean_document = [token for token in clean_document if token not in stoplist]
    label = row.Y02
    importance = row.Y02_imp
    ind = row.APPLN_ID

    clean_document = row.LEMMAS
    # Remove some additional stopwords
    clean_document = [token for token in clean_document if token not in stoplist]
    # Remove Y04 and Y10 tag
    labels = [cpc for cpc in row.CPC if cpc not in ['Y04', 'Y10']]
        
    # Create word embedding matrix
    patent_embedding = word_list_to_embedding_array(clean_document)
    len_patent_embedding = len(patent_embedding)
    
    # Calculate proximity to all clean technology semantic spaces
    for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
        for n_words in [10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000, 3000, 4000]:
            for embedding_type in ['we', 'we_avg', 'bert']:
                if embedding_type == 'we':
                    technology_embedding = get_semantic_vectors(y02, n_words)

                    # Calculate cosine similarity between all permutations of patent vector space and technology semantic vector space
                    similarity = np.round_(cosine_similarity(patent_embedding, technology_embedding).flatten(), decimals=5)
                    similarity[similarity < 0] = 0
                    similarity_mean = similarity.mean()
                    # Calculate mean embedding and then cosine similarity between both document embeddings
                    similarity_mean2 = cosine_similarity_vectors(patent_embedding.mean(axis=0), technology_embedding.mean(axis=0))
                    similarity_mean2[similarity_mean2 < 0] = 0

                    # Calculate number of exact word matches
                    n_exact = (similarity == 1).sum()
                    n_exact_norm = n_exact/len_patent_embedding

                    temp.append([ind, label, y02, importance, n_words, similarity_mean, similarity_mean2, n_exact, n_exact_norm, n_exact_norm+similarity_mean, n_exact_norm+similarity_mean2])
    #if index==5:
    #    break
            
df_prox = pd.DataFrame(temp, columns=['APPLN_ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN', 'MEAN2', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN', 'N_EXACT_NORM_MEAN2'])

# +
stoplist = ['and/or', '/h', 't1', 'dc', 'mm', 'wt', '113a', '115a', 'ofdm', 'lpwa']
temp = []
for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], position=0, leave=True):
#for index in tqdm(range(df_test.shape[0]), position=0, leave=True):
    #row = df_test.iloc[index]
    clean_document = row.ABSTRACT.split()
    # Remove some additional stopwords
    clean_document = [token for token in clean_document if token not in stoplist]
    label = row.Y02
    importance = row.Y02_imp
    ind = row.APPLN_ID

    clean_document = row.LEMMAS
    # Remove some additional stopwords
    clean_document = [token for token in clean_document if token not in stoplist]
    # Remove Y04 and Y10 tag
    labels = [cpc for cpc in row.CPC if cpc not in ['Y04', 'Y10']]
        
    # Create word embedding matrix
    patent_embedding = word_list_to_embedding_array(clean_document)
    len_patent_embedding = len(patent_embedding)
    
    # Calculate proximity to all clean technology semantic spaces
    for y02 in ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']:
        for n_words in [10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000, 3000, 4000]:
            technology_embedding = get_semantic_vectors(y02, n_words)
            
            # Calculate cosine similarity between all permutations of patent vector space and technology semantic vector space
            similarity = np.round_(cosine_similarity(patent_embedding, technology_embedding).flatten(), decimals=5)
            similarity[similarity < 0] = 0
            similarity_mean = similarity.mean()
            # Calculate mean embedding and then cosine similarity between both document embeddings
            similarity_mean2 = cosine_similarity_vectors(patent_embedding.mean(axis=0), technology_embedding.mean(axis=0))
            similarity_mean2[similarity_mean2 < 0] = 0
        
            # Calculate number of exact word matches
            n_exact = (similarity == 1).sum()
            n_exact_norm = n_exact/len_patent_embedding
        
            temp.append([ind, label, y02, importance, n_words, similarity_mean, similarity_mean2, n_exact, n_exact_norm, n_exact_norm+similarity_mean, n_exact_norm+similarity_mean2])
    #if index==5:
    #    break
            
df_prox = pd.DataFrame(temp, columns=['APPLN_ID', 'LABEL', 'Y02', 'Y02_IMPORTANCE', 'N_WORDS', 'MEAN', 'MEAN2', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN', 'N_EXACT_NORM_MEAN2'])
# -

df_prox

# Save to disk
df_prox.to_csv(here(r'.\03_Model\temp\df_validation_patents.txt'), sep='\t', encoding='utf-8')

# + [markdown] tags=[]
# ## Some visualizations 
# -

# Load testing file
df_test_results = pd.read_csv(here(r'.\03_Model\temp\df_validation_patents.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')

df_test_results.head(3)

df_test_results_l = pd.melt(df_test_results, id_vars=['LABEL', 'N_WORDS', ], value_vars=['MEAN', 'MEAN2', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN', 'N_EXACT_NORM_MEAN2'], var_name='measure', value_name='value')

df_test_results_l

sns.catplot(
    data=df_test_results_l, 
    x="N_WORDS", y="value",
    hue="LABEL",  col="measure", kind="box",
    col_wrap=2, sharey=False, sharex=True, height=6, aspect=1.5
)

# It appears that all proximity measures allow a differentiation between cleantech and non-cleantech patents (with the last one = sum over fraction of exact matches and mean cosine similarity between semantic vector spaces showing this differentiation best.) Nonetheless differentiation could be better. Ideas for improvement clearly exist:
# - training own technology-related word embeddings []
# - lemmatization before generating the the semantic vector spaces for the different technology classes [x]
#
# In any case: The test data can also be used as validation set in order to determine the "best" number of words to define the semantic technology space.

# Add Y02_dict to test data
#df_test = df_test_results.merge(df_test[['APPLN_ID', 'Y02_dict']], how='left', left_on='APPLN_ID', right_on='APPLN_ID')
df_test['APPLN_ID'] = df_test.APPLN_ID.astype(np.int64)
df_test_results = df_test_results.merge(df_test[['APPLN_ID', 'Y02_dict']], how='left', left_on='APPLN_ID', right_on='APPLN_ID')

df_test_results['Y02_true'] = df_test_results.Y02_dict.apply(lambda x: list(x.keys()))

# Analyze how proximity differs compared to non-cleantech patents.

# Check if color map is loaded
greens_dict

# +
fig, axes = plt.subplots(figsize=(15,20), ncols=2, nrows=4)

y02s = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']

for y02, ax in zip(y02s, axes.flat):
    df_y02 = df_test_results.loc[df_test_results.Y02_true.apply(lambda x: y02 in x) & (df_test_results.Y02 == y02)].copy()
    df_y02['LABEL'] = y02
    N_y02 = len(df_y02['APPLN_ID'].drop_duplicates())
    df_nony02 = df_test_results.loc[(df_test_results.LABEL == 0) & (df_test_results.Y02 == y02)].copy()
    df_nony02['LABEL'] = 'non-cleantech'
    N_nony02 = len(df_nony02['APPLN_ID'].drop_duplicates())
    df_temp = pd.concat([df_y02, df_nony02])
    df_temp["TECH_PROX"] = df_temp.N_EXACT_NORM_MEAN/2
    palette = {'non-cleantech': 'lightgrey', y02: greens_dict[y02]}
    ax = sns.boxplot(x="N_WORDS", y="TECH_PROX", hue="LABEL",
                     data=df_temp, linewidth=1, ax=ax, palette=palette
                     #,showmeans=True, meanprops={"marker":"o",
                     #                           "markerfacecolor":"white", 
                     #                           "markeredgecolor":"black",
                     #                           "markersize":"5"}
                    )
    ax.set_title('Proximity to ' + y02 + ' (based on $N_{' + y02 + '}$ = ' + str(N_y02) + ', ' + '$N_{non-y02}$ = ' + str(N_nony02) + ')')
    ax.set_xlabel('Size of semantic technology space (number of words)')
    ax.set_ylabel('Technological proximity')
    ax.legend_.set_title('Patent class')
plt.suptitle("Technological proximity based on out-of sample patents", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# -

# Analyze how proximity differs compared to cleantech patents from other Y02 classes.

# +
fig, axes = plt.subplots(figsize=(15,20), ncols=2, nrows=4)

y02s = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']

for y02, ax in zip(y02s, axes.flat):
    df_y02 = df_test_results.loc[df_test_results.Y02_true.apply(lambda x: y02 in x) & (df_test_results.Y02 == y02)].copy()
    df_y02['LABEL'] = y02
    N_y02 = len(df_y02['APPLN_ID'].drop_duplicates())
    df_othery02 = df_test_results.loc[df_test_results.Y02_true.apply(lambda x: (y02 not in x) & (len(x)>0)) & (df_test_results.Y02 == y02)].copy()
    df_othery02['LABEL'] = 'other cleantech'
    N_othery02 = len(df_othery02['APPLN_ID'].drop_duplicates())
    df_temp = pd.concat([df_y02, df_othery02])
    df_temp["TECH_PROX"] = df_temp.N_EXACT_NORM_MEAN/2
    palette = {'other cleantech': 'lightgrey', y02: greens_dict[y02]}
    ax = sns.boxplot(x="N_WORDS", y="TECH_PROX", hue="LABEL",
                 data=df_temp, linewidth=1, ax=ax, palette=palette)
    ax.set_title('Proximity to ' + y02 + ' (based on $N_{' + y02 + '}$ = ' + str(N_y02) + ', ' + '$N_{other-y02}$ = ' + str(N_othery02) + ')')
    ax.set_xlabel('Size of semantic technology space (number of words)')
    ax.set_ylabel('Technological proximity')
    ax.legend_.set_title('Patent class')
plt.suptitle("Technological proximity based on out-of sample patents", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# -

# Results look good.

# + [markdown] tags=[]
# # Testing on corporate websites 
# -

from util import string_to_lemma
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

# Careful: length texts is not equally distributed between cleantech firms and NASDAQ firms!

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
        for n_words in [10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000, 3000, 4000]:
            technology_embedding = get_semantic_vectors(y02, n_words)
            
            # Calculate cosine similarity between all permutations of patent vector space and technology semantic vector space
            similarity = np.round_(cosine_similarity(patent_embedding, technology_embedding).flatten(), decimals=5)
            similarity[similarity < 0] = 0
            similarity_mean = similarity.mean()
            # Calculate mean embedding and then cosine similarity between both document embeddings
            similarity_mean2 = cosine_similarity_vectors(patent_embedding.mean(axis=0), technology_embedding.mean(axis=0))
            similarity_mean2[similarity_mean2 < 0] = 0
        
            # Calculate number of exact word matches
            n_exact = (similarity == 1).sum()
            n_exact_norm = n_exact/len_patent_embedding
        
            temp.append([ind, label, y02, n_words, similarity_mean, similarity_mean2, n_exact, n_exact_norm, n_exact_norm+similarity_mean, n_exact_norm+similarity_mean2])
    #if index==5:
    #    break
            
df_prox = pd.DataFrame(temp, columns=['APPLN_ID', 'LABEL', 'Y02', 'N_WORDS', 'MEAN', 'MEAN2', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN', 'N_EXACT_NORM_MEAN2'])
# -

df_firms = pd.melt(df_firms_results, id_vars=['LABEL', 'N_WORDS', ], value_vars=['MEAN', 'N_EXACT', 'N_EXACT_NORM', 'N_EXACT_NORM_MEAN'], var_name='measure', value_name='value')

sns.catplot(
    data=df_firms,
    x="N_WORDS", y="value",
    hue="LABEL",  col="measure", kind="box",
    col_wrap=1, sharey=False, sharex=True, height=6, aspect=2
)

# +
fig, axes = plt.subplots(figsize=(15,20), ncols=2, nrows=4)

y02s = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']

for y02, ax in zip(y02s, axes.flat):
    df_temp = df_firms_results.loc[df_firms_results.Y02 == y02].copy()
    df_temp["TECH_PROX"] = df_temp.N_EXACT_NORM_MEAN/2
    palette = {'nasdaq': 'lightgrey', 'cleantech': greens[len(greens)-1]}
    ax = sns.boxplot(x="N_WORDS", y="TECH_PROX", hue="LABEL",
                 data=df_temp, linewidth=1, ax=ax, palette=palette)
    ax.set_title('Proximity to ' + y02)
    ax.set_xlabel('Size of semantic technology space (number of words)')
    ax.set_ylabel('Technological proximity')
    ax.legend_.set_title('Company type')
    new_labels = ['Cleantech-100', 'NASDAQ-100']
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)
plt.suptitle("Technological proximity based on company descriptions", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# -

# Discrimination between company descriptions looks promising, too!
