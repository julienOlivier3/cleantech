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
from pyprojroot import here
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from util import string_to_lemma, TP, TN, FP, FN, ACC, PRECISION, RECALL, F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
#model = SentenceTransformer('msmarco-distilbert-base-v4')
#model._first_module().max_seq_length = 510 # increase maximum sequence length which is 128 by default

# Read the scraped firm-level data of cleantech companies.

df = pd.read_csv(here('01_Data/02_Firms/df_cleantech_firms.txt'), sep='\t')

# Drop duplicate firms (i.e. firms which made it to the Cleantech-100 list in subsequent years)
df = df.drop_duplicates('ID')
# Drop row with missings only
df = df.loc[df.ID.notnull(),:]

df_sp = pd.read_csv(here('01_Data/02_Firms/df_all_sp_firms.txt'), sep='\t')

# Drop companies without business description
df_sp = df_sp.loc[df_sp.BUSINESS_SUMMARY.notnull()]

# Check if there are cleantech firms listed on S&P 500
drop = set(df_sp.SYMBOL.values).intersection(set(df.TICKER_SYMBOL.drop_duplicates().values))
drop

# Drop cleantech companies listed on S&P 500
df_sp = df_sp.loc[~df_sp.SYMBOL.isin(drop)]

# Read the mapping from tags to cleantech markets.

df_map = pd.read_excel(here('./01_Data/02_Firms/df_cleantech_firms_label.xlsx'), sheet_name='Tag2CPC')

df_map = df_map.loc[df_map.PRIMARY_TAG.notnull() & df_map.ABBREVIATION.notnull(), ['PRIMARY_TAG', 'ABBREVIATION']].drop_duplicates()
tag2market = dict(zip(df_map.PRIMARY_TAG, df_map.ABBREVIATION))

# Note that energy storage is mapped to E-Mobility and Grid. To solve this double mapping, I assign energy storage to E-Mobility if the company is additionally tagged with 'transportation & logistics'.

df.loc[(df.PRIMARY_TAG=='energy storage') & df.TAGS.apply(lambda x: 'transportation & logistics' in x), 'PRIMARY_TAG'] = 'energy storage transportation'

# Generate cleantech market class
df['CLEANTECH_MARKET'] = df.PRIMARY_TAG.map(tag2market)

# Note that the additional tags shall be used to get a better classification.

df.loc[df.TAGS.apply(lambda x: 'decarbonisation' in x), ['NAME', 'SHORT_DESCRIPTION', 'PRIMARY_TAG', 'TAGS']].style

df.loc[df.PRIMARY_TAG=='air', 'CLEANTECH_MARKET'] = 'Generation'
df.loc[df.TAGS.apply(lambda x: any(i for i in ['carbon capture and sequestration (ccs)', 'carbon capture utilization (ccu)', 'carbon capture utilization and storage (ccus)'] if i in x)), 'CLEANTECH_MARKET'] = 'CCS'

# Reduce to those companies for which cleantech market assignment worked (companies with 'other cleantech' tag are dropped)
df = df.loc[df.CLEANTECH_MARKET.notnull(), ['ID', 'NAME', 'PRIMARY_TAG', 'TAGS', 'SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW', 'CLEANTECH_MARKET']]


# + [markdown] tags=[]
# ## Embeddings
# -

# This approach calculates company and technology embeddings and cosine similarity as proximity measure between company and technology.

# Define some functions required to conduct the technological proximity calculation.

# Text concatenator
def create_joint_string(x, columns = ['SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW']):
    return(' '.join([i for i in list(x[columns].values) if not pd.isnull(i)]))    


# Company Embeddings
def company2embedding(df, model, columns=['SHORT_DESCRIPTION'], stop_words = ['developer'], lemmatize=True):
    
    # Text concatenation
    df['DESC'] = df.apply(lambda x: create_joint_string(x, columns=columns), axis=1)
    
    company = df.ID.values
    technology = df.CLEANTECH_MARKET.values
    
    # Lemmatization, stop word removal and sentence embedding
    if lemmatize:
        # Conduct lemmatization
        df['DESC'] = df.DESC.apply(lambda x: string_to_lemma(x))
        # Remove stop words
        df['DESC'] = df.DESC.apply(lambda x: [word for word in x if word not in stop_words])
        # Concatenate list of words to whitespace seperated string
        df['DESC'] = df.DESC.apply(lambda x: ' '.join(str(i) for i in x))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.DESC.values)
    
    # Stop word removal Sentence embedding
    else:
        # Remove stop words
        df['DESC'] = df.DESC.apply(lambda x: ' '.join(word.lower() for word in x.split() if word.lower() not in stop_words))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.DESC.values)

    return(company, technology, embeddings)


# Company Embeddings (all)
def allcompany2embedding(df1, df2, model, columns=['SHORT_DESCRIPTION'], stop_words = [], lemmatize=True):
    
    # Text concatenation
    df1['ID'] = df1.ID.astype(int).astype(str)
    df1['CLEANTECH'] = 1
    df1['DESC'] = df1.apply(lambda x: create_joint_string(x, columns=columns), axis=1)
    df1 = df1[['ID', 'CLEANTECH', 'CLEANTECH_MARKET', 'DESC']]
    
    # df2['ID'] = range((len(df_nasdaq)+1)*-1, -1)
    # df2['CLEANTECH'] = 0
    # df2['CLEANTECH_MARKET'] = 'None'
    # df2.rename(columns={'BUSINESS_SUMMARY': 'DESC'}, inplace=True)
    # df2 = df2.loc[df2.DESC.notnull(), ['ID', 'CLEANTECH', 'CLEANTECH_MARKET', 'DESC']]
    
    df2 = df2.loc[df2.BUSINESS_SUMMARY.notnull()].copy()
    df2['ID'] = df2.SYMBOL
    df2['CLEANTECH'] = 0
    df2['CLEANTECH_MARKET'] = 'None'
    df2.rename(columns={'BUSINESS_SUMMARY': 'DESC'}, inplace=True)
    df2 = df2[['ID', 'CLEANTECH', 'CLEANTECH_MARKET', 'DESC']]
    
    df = pd.concat([df1, df2])
    
    company = df.ID.values
    cleantech = df.CLEANTECH.values
    technology = df.CLEANTECH_MARKET.values
    
    # Lemmatization, stop word removal and sentence embedding
    if lemmatize:
        # Conduct lemmatization
        df['DESC'] = df.DESC.apply(lambda x: string_to_lemma(x))
        # Remove stop words
        df['DESC'] = df.DESC.apply(lambda x: [word for word in x if word not in stop_words])
        # Concatenate list of words to whitespace seperated string
        df['DESC'] = df.DESC.apply(lambda x: ' '.join(str(i) for i in x))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.DESC.values, show_progress_bar=True)
    
    # Stop word removal Sentence embedding
    else:
        # Remove stop words
        df['DESC'] = df.DESC.apply(lambda x: ' '.join(word.lower() for word in x.split() if word.lower() not in stop_words))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.DESC.values, show_progress_bar=True)

    return(company, cleantech, technology, embeddings)


# Read topic-proba-df
df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words_markets.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')


# Technology Embedding
def technology2embedding(df, model, technology, n_words):
    semantic_tech = ' '.join(str(word).lower() for word in list(df.loc[df.Topic==technology].head(n_words).Word.values))
    embedding = model.encode(semantic_tech).reshape(1, -1)

    return(embedding)


# Now calculate similarity between company embeddings and technology embeddings. 

# Embeddings and cosine similarity
tech_proxs = []
c, c_tech, c_emb = company2embedding(df, model, columns=['SHORT_DESCRIPTION'], stop_words=[], lemmatize=False)
techs = list(set(c_tech))
n_words = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 75, 100, 200, 250, 300, 400]
for tech in tqdm(techs):
    tech_id = np.zeros(shape=(len(c_tech)))
    tech_id[np.where(c_tech == tech)] = 1
    for n_word in n_words:
        t_emb = technology2embedding(df_topic_words, model, technology=tech, n_words=n_word)
        tech_prox = cosine_similarity(c_emb, t_emb).reshape(len(c_emb),)
        tech_prox[tech_prox < 0] = 0
        tech_proxs.append(pd.DataFrame({'COMPANY': c, 'TECHNOLOGY': tech, 'TECHNOLOGY_ID': tech_id, 'TECHNOLOGY_PROXIMITY': tech_prox, 'N_WORDS': n_word}))
        df_prox = pd.concat(tech_proxs)

# Visualize results.

# +
fig, axes = plt.subplots(figsize=(15,20), ncols=2, nrows=5)

for tech, ax in zip(techs, axes.flat):
    df_tech = df_prox.loc[df_prox.TECHNOLOGY == tech].copy()
    N_true = len(df_tech.loc[df_tech.TECHNOLOGY_ID==1, 'COMPANY'].drop_duplicates())
    N_false = len(df_tech.loc[df_tech.TECHNOLOGY_ID==0, 'COMPANY'].drop_duplicates())
    palette = {0: 'lightgrey', 1: 'green'}
    ax = sns.boxplot(x="N_WORDS", y="TECHNOLOGY_PROXIMITY", hue="TECHNOLOGY_ID",
                     data=df_tech, linewidth=1, ax=ax, palette=palette
                     #,showmeans=True, meanprops={"marker":"o",
                     #                           "markerfacecolor":"white", 
                     #                           "markeredgecolor":"black",
                     #                           "markersize":"5"}
                    )
    ax.set_title('Proximity to ' + tech + ' (based on $N_{' + tech + '}$ = ' + str(N_true) + ', ' + '$N_{non-' + tech + '}$ = ' + str(N_false) + ')')
    ax.set_xlabel('Size of semantic technology space (number of words)')
    ax.set_ylabel('Technological proximity')
    ax.legend_.set_title('Clean technology')
plt.suptitle("Technological proximity based on labeled cleantech firms", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# -

# Calculate classification metrics.

# + tags=[]
class_metrics = []

for tech in techs:
    for n_word in n_words:
        df_temp = df_prox.loc[(df_prox.TECHNOLOGY==tech) & (df_prox.N_WORDS==n_word)]
        true = df_temp.TECHNOLOGY_ID.values
        prox = df_temp.TECHNOLOGY_PROXIMITY.values
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred = prox >= th
            
            class_metrics.append([
                tech, n_word, th, 
                ACC(true, pred),
                RECALL(true, pred),
                PRECISION(true, pred),
                F(true, pred, beta=1)])
            
df_metric = pd.DataFrame(class_metrics, columns=['TECHNOLOGY', 'N_WORDS', 'THRESHHOLD', 
                                                 'ACCURACY', 'RECALL', 'PRECISION', 'F-SCORE'])
# -

df_metric.loc[df_metric.groupby('TECHNOLOGY')['F-SCORE'].idxmax()]

# Now do the same calculations but turn it into a binary classification differentiating between cleantech and non-cleantech firm.

c, clean, c_tech, c_emb = allcompany2embedding(df, df_sp, model, columns=['SHORT_DESCRIPTION'], stop_words = [], lemmatize=False)

# + tags=[] active=""
# # Save results
# c_embs = {'c': c, 'clean': clean, 'c_tech': c_tech, 'c_emb': c_emb}
# with open(here('./01_Data/02_Firms/company_embeddings.pkl'), 'wb') as f:
#     pkl.dump(c_embs, f)

# + tags=[]
# Embeddings and cosine similarity
tech_proxs = []
techs = [i for i in list(set(c_tech)) if i not in ['None', np.nan]]
n_words = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 75, 100, 200, 250, 300, 400]
for tech in tqdm(techs):
    tech_id = np.zeros(shape=(len(c_tech)))
    tech_id[np.where(c_tech == tech)] = 1
    tech_id[np.where(c_tech == 'None')] = 2
    clean_id = np.zeros(shape=(len(clean)))
    clean_id[np.where(clean == 1)] = 1
    for n_word in n_words:
        t_emb = technology2embedding(df_topic_words, model, technology=tech, n_words=n_word)
        tech_prox = cosine_similarity(c_emb, t_emb).reshape(len(c_emb),)
        tech_prox[tech_prox < 0] = 0
        tech_proxs.append(pd.DataFrame({'COMPANY': c, 'CLEANTECH': clean_id, 'TECHNOLOGY': tech, 'TECHNOLOGY_ID': tech_id, 'TECHNOLOGY_PROXIMITY': tech_prox, 'N_WORDS': n_word}))
        df_prox = pd.concat(tech_proxs)

# +
fig, axes = plt.subplots(figsize=(15,20), ncols=2, nrows=5)

for tech, ax in zip(techs, axes.flat):
    df_tech = df_prox.loc[df_prox.TECHNOLOGY == tech].copy()
    N_true = len(df_tech.loc[df_tech.TECHNOLOGY_ID==1, 'COMPANY'].drop_duplicates())
    N_false = len(df_tech.loc[df_tech.TECHNOLOGY_ID==0, 'COMPANY'].drop_duplicates())
    palette = {0: 'lightgrey', 1: 'green', 2: 'red'}
    ax = sns.boxplot(x="N_WORDS", y="TECHNOLOGY_PROXIMITY", hue="TECHNOLOGY_ID",
                     data=df_tech, linewidth=1, ax=ax, palette=palette
                     #,showmeans=True, meanprops={"marker":"o",
                     #                           "markerfacecolor":"white", 
                     #                           "markeredgecolor":"black",
                     #                           "markersize":"5"}
                    )
    ax.set_title('Proximity to ' + tech + ' (based on $N_{' + tech + '}$ = ' + str(N_true) + ', ' + '$N_{non-' + tech + '}$ = ' + str(N_false) + ')')
    ax.set_xlabel('Size of semantic technology space (number of words)')
    ax.set_ylabel('Technological proximity')
    ax.legend_.set_title('Clean technology')
plt.suptitle("Technological proximity based on labeled cleantech firms", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# +
class_metrics = []

for tech in techs:
    for n_word in n_words:
        df_temp = df_prox.loc[(df_prox.TECHNOLOGY==tech) & (df_prox.N_WORDS==n_word)]
        
        true1 = df_temp.loc[df_temp.TECHNOLOGY_ID.isin([0,1])].TECHNOLOGY_ID.values
        true2 = df_temp.loc[df_temp.TECHNOLOGY_ID.isin([1,2])].CLEANTECH.values
        prox1 = df_temp.loc[df_temp.TECHNOLOGY_ID.isin([0,1])].TECHNOLOGY_PROXIMITY.values
        prox2 = df_temp.loc[df_temp.TECHNOLOGY_ID.isin([1,2])].TECHNOLOGY_PROXIMITY.values
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred1 = prox1 >= th
            pred2 = prox2 >= th
            
            
            class_metrics.append([
                tech, n_word, th, 
                ACC(true1, pred1),
                RECALL(true1, pred1),
                PRECISION(true1, pred1),
                F(true1, pred1, beta=1),
                len(true1),
                ACC(true2, pred2),
                RECALL(true2, pred2),
                PRECISION(true2, pred2),
                F(true2, pred2, beta=1),
                len(true2)
            ])
            
df_metric = pd.DataFrame(class_metrics, columns=['TECHNOLOGY', 'N_WORDS', 'THRESHHOLD', 
                                                 'ACCURACY', 'RECALL', 'PRECISION', 'F-SCORE', 'N',
                                                 'ACCURACY_nc', 'RECALL_nc', 'PRECISION_nc', 'F-SCORE_nc', 'N_nc'])
# -

df_metric.loc[df_metric.groupby('TECHNOLOGY')['F-SCORE_nc'].idxmax()]

# Finally, check the binary case.

# Do a dev-test split to conduct tuning of hyperparameters n_words and threshold and then evaluate performance on test set.

# +
# Create dev and test data
np.random.seed(3)
dev_fraction = 0.5
company_id = list(df_prox.COMPANY.drop_duplicates().values)
dev_id = list(np.random.choice(company_id, size=int(np.round(dev_fraction*len(company_id))), replace=False))
test_id = list(set(company_id).difference(set(dev_id)))

df_dev = df_prox.loc[df_prox.COMPANY.isin(dev_id)]
df_test = df_prox.loc[df_prox.COMPANY.isin(test_id)]

# +
# Tune hyperparameters on the dev set using grid search
class_metrics = []

for n_word in n_words:
    df_temp = df_dev.loc[(df_dev.N_WORDS==n_word)].groupby(['COMPANY', 'CLEANTECH']).agg({'TECHNOLOGY_PROXIMITY': max}).reset_index().merge(df_dev.loc[(df_dev.N_WORDS==n_word)].groupby(['COMPANY', 'CLEANTECH', 'TECHNOLOGY']).agg({'TECHNOLOGY_PROXIMITY': max}).reset_index())
        
    true = df_temp.CLEANTECH.values
    prox = df_temp.TECHNOLOGY_PROXIMITY.values
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = prox >= th            
            
        class_metrics.append([
                n_word, th, 
                ACC(true, pred),
                RECALL(true, pred),
                PRECISION(true, pred),
                F(true, pred, beta=1),
                sum(true),
                len(true) - sum(true)
            ])
            
df_metric = pd.DataFrame(class_metrics, columns=['N_WORDS', 'THRESHHOLD', 'ACCURACY', 'RECALL', 'PRECISION', 'F-SCORE', 'N_CLEANTECH', 'N_NONE'])
# -

df_metric.loc[df_metric['F-SCORE'].idxmax()]

# +
# Tune hyperparameters on the dev set using grid search
class_metrics = []
n_word = 9
th = 0.3
df_temp = df_test.loc[(df_test.N_WORDS==n_word)].groupby(['COMPANY', 'CLEANTECH']).agg({'TECHNOLOGY_PROXIMITY': max}).reset_index().merge(df_test.loc[(df_test.N_WORDS==n_word)].groupby(['COMPANY', 'CLEANTECH', 'TECHNOLOGY']).agg({'TECHNOLOGY_PROXIMITY': max}).reset_index())
        
true = df_temp.CLEANTECH.values
prox = df_temp.TECHNOLOGY_PROXIMITY.values
pred = prox >= th            
            
class_metrics.append([
                n_word, th, 
                ACC(true, pred),
                RECALL(true, pred),
                PRECISION(true, pred),
                F(true, pred, beta=1),
                sum(true),
                len(true) - sum(true)
            ])
            
df_metric = pd.DataFrame(class_metrics, columns=['N_WORDS', 'THRESHHOLD', 'ACCURACY', 'RECALL', 'PRECISION', 'F-SCORE', 'N_CLEANTECH', 'N_NONE'])
# -

df_metric

# Check false positives.

temp = df_prox.loc[df_prox.CLEANTECH==0].groupby('COMPANY').agg({'TECHNOLOGY_PROXIMITY': max}).sort_values('TECHNOLOGY_PROXIMITY', ascending=False)
temp

df_sp.loc[df_sp.SYMBOL.isin(temp.head(5).index), ['SYMBOL', 'SECURITY', 'GICS_SUB_INDUSTRY', 'BUSINESS_SUMMARY']].style


# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Semantic Overlap 
# -

# This approach uses a Jaccard-kind similarity index based on an overlap in words between semantic technology description and company description.

# Company words
def company2wordlist(df, columns=['SHORT_DESCRIPTION'], stop_words = ['developer']):
    # Text concatenation
    df['DESC'] = df.apply(lambda x: create_joint_string(x, columns=columns), axis=1)
    
    company = df.ID.values
    technology = df.CLEANTECH_MARKET.values
    
    # Lemmatization and stop word removal
    # Conduct lemmatization
    df['DESC'] = df.DESC.apply(lambda x: string_to_lemma(x))
    # Remove stop words
    df['DESC'] = df.DESC.apply(lambda x: [word for word in x if word not in stop_words])
    
    wordlist = df.DESC.values
    
    return(company, technology, wordlist)


# Technology words
def technology2wordlist(df, technology, n_words):
    wordlist = [str(word).lower() for word in list(df.loc[df.Topic==technology].head(n_words).Word.values)]
    
    return(wordlist)


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    #union = (len(set(list1)) + len(set(list2))) - intersection
    union = min(len(set(list1)), len(set(list2)))
    return float(intersection) / union


# Semantic overlap
tech_proxs = []
c, c_techs, c_lists = company2wordlist(df, columns=['SHORT_DESCRIPTION'], stop_words=[])
techs = list(set(c_techs))
n_words = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 75, 100, 200, 250, 300, 400]
for tech in tqdm(techs):
    tech_id = np.zeros(shape=(len(c_techs)))
    tech_id[np.where(c_techs == tech)] = 1
    for n_word in n_words:
        t_list = technology2wordlist(df_topic_words, technology=tech, n_words=n_word)
        tech_prox = []
        for c_list in c_lists:
            tech_prox.append(jaccard_similarity(c_list, t_list))
        tech_proxs.append(pd.DataFrame({'COMPANY': c, 'TECHNOLOGY': tech, 'TECHNOLOGY_ID': tech_id, 'TECHNOLOGY_PROXIMITY': tech_prox, 'N_WORDS': n_word}))
        df_prox = pd.concat(tech_proxs)

# +
fig, axes = plt.subplots(figsize=(15,20), ncols=2, nrows=5)

for tech, ax in zip(techs, axes.flat):
    df_tech = df_prox.loc[df_prox.TECHNOLOGY == tech].copy()
    N_true = len(df_tech.loc[df_tech.TECHNOLOGY_ID==1, 'COMPANY'].drop_duplicates())
    N_false = len(df_tech.loc[df_tech.TECHNOLOGY_ID==0, 'COMPANY'].drop_duplicates())
    palette = {0: 'lightgrey', 1: 'green'}
    ax = sns.boxplot(x="N_WORDS", y="TECHNOLOGY_PROXIMITY", hue="TECHNOLOGY_ID",
                     data=df_tech, linewidth=1, ax=ax, palette=palette
                     #,showmeans=True, meanprops={"marker":"o",
                     #                           "markerfacecolor":"white", 
                     #                           "markeredgecolor":"black",
                     #                           "markersize":"5"}
                    )
    ax.set_title('Proximity to ' + tech + ' (based on $N_{' + tech + '}$ = ' + str(N_true) + ', ' + '$N_{non-' + tech + '}$ = ' + str(N_false) + ')')
    ax.set_xlabel('Size of semantic technology space (number of words)')
    ax.set_ylabel('Technological proximity')
    ax.legend_.set_title('Clean technology')
plt.suptitle("Technological proximity based on labeled cleantech firms", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# + tags=[]
class_metrics = []

for tech in techs:
    for n_word in n_words:
        df_temp = df_prox.loc[(df_prox.TECHNOLOGY==tech) & (df_prox.N_WORDS==n_word)]
        true = df_temp.TECHNOLOGY_ID.values
        prox = df_temp.TECHNOLOGY_PROXIMITY.values
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred = prox >= th
            
            class_metrics.append([
                tech, n_word, th, 
                ACC(true, pred),
                RECALL(true, pred),
                PRECISION(true, pred),
                F(true, pred, beta=1)])
        df_metric = pd.DataFrame(class_metrics, columns=['TECHNOLOGY', 'N_WORDS', 'THRESHHOLD', 
                                                         'ACCURACY', 'RECALL', 'PRECISION', 'F-SCORE'])
# -

df_metric.loc[df_metric.groupby('TECHNOLOGY')['F-SCORE'].idxmax()]


# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Semantic search
# -

# This approach uses a semantic search approach. For each embedded company description the x closest patent embeddings are found from a corpus of cleantech patents.

# First, create company embeddings.

# Text concatenator
def create_joint_string(x, columns = ['SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW']):
    return(' '.join([i for i in list(x[columns].values) if not pd.isnull(i)]))    


# Company Embeddings
def company2embedding(df, model, columns=['SHORT_DESCRIPTION'], stop_words = ['developer'], lemmatize=True):
    
    # Text concatenation
    df['DESC'] = df.apply(lambda x: create_joint_string(x, columns=columns), axis=1)
    
    company = df.ID.values
    technology = df.CLEANTECH_MARKET.values
    
    # Lemmatization, stop word removal and sentence embedding
    if lemmatize:
        # Conduct lemmatization
        df['DESC'] = df.DESC.apply(lambda x: string_to_lemma(x))
        # Remove stop words
        df['DESC'] = df.DESC.apply(lambda x: [word for word in x if word not in stop_words])
        # Concatenate list of words to whitespace seperated string
        df['DESC'] = df.DESC.apply(lambda x: ' '.join(str(i) for i in x))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.DESC.values)
    
    # Stop word removal Sentence embedding
    else:
        # Remove stop words
        df['DESC'] = df.DESC.apply(lambda x: ' '.join(word.lower() for word in x.split() if word.lower() not in stop_words))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.DESC.values)

    return(company, technology, embeddings)


c, c_tech, c_emb = company2embedding(df, model, columns=['SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW'], stop_words=[], lemmatize=False)

# Now create patent embeddings.

# Read patent data
df_pat = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))
df_pat.shape

# Create column that contains cleantech labels only
techs = list(set(c_tech))
df_pat['CLEANTECH_MARKET'] = df_pat.CLEANTECH_MARKET.apply(lambda x: list(set(x).intersection(set(techs))))

# Reduce to patents which have been exclusively assigned to a cleantech label
df_pat = df_pat.loc[df_pat.CLEANTECH_MARKET.apply(len)>0]
df_pat.shape

# Read embedded abstracts of the above patents
with open(here(r'.\01_Data\01_Patents\patent_embeddings.pkl'), 'rb') as f:
    p_embs = pkl.load(f)
p_embs.shape

df_pat = df_pat.reset_index(drop=True)

# Create label column with NaN for those patents which are not uniquely assigned to just one cleantech label
df_pat.loc[df_pat.CLEANTECH_MARKET.apply(len)==1, 'LABEL'] = df_pat.loc[df_pat.CLEANTECH_MARKET.apply(len)==1].CLEANTECH_MARKET.apply(lambda x: x[0])

df_pat.LABEL.value_counts(dropna=False)

# Create corpus of patents for semantic search balanced by cleantech label
p_corpus = df_pat.groupby('LABEL').sample(640)

# Reduce embedding array respectively
index = p_corpus.index
p = df_pat.loc[df_pat.index.isin(index), 'APPLN_ID'].values
p_tech = df_pat.loc[df_pat.index.isin(index), 'LABEL'].values
p_emb = p_embs[index]


def convert2df(top_overlap, company_id, company_label, patent_id, patent_label):
    df_prox=[]
    for ind, company in enumerate(top_overlap):
        temp=pd.DataFrame(company)
        temp['APPLN_ID'] = patent_id[temp.corpus_id.values]
        temp['P_LABEL'] = patent_label[temp.corpus_id.values]
        temp['ID'] = company_id[ind]
        temp['C_LABEL'] = company_label[ind]
        df_prox.append(temp)
    df_prox=pd.concat(df_prox)
    #df_prox=df_prox.groupby('corpus_id').agg({'score': np.median}).merge(df_prox.groupby('corpus_id').count().rename(columns={'score': 'n'})/len(s_search), left_index=True, right_index=True)
    #df_prox['similarity'] = (df_prox.score+df_prox.n)/2
    #df_prox = df_prox.sort_values('similarity', ascending=False).reset_index().rename(columns={'corpus_id': 'ID'})
    return(df_prox)


def embeddings2similarity(query_embeddings, company_id, company_label, corpus_embeddings, patent_id, patent_label,  top_k=5):
    top_overlap = semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)
    
    return(convert2df(top_overlap, company_id, company_label, patent_id, patent_label))


df_prox = embeddings2similarity(c_emb, c, c_tech, p_emb, p, p_tech)

df.loc[df.ID==381917.0].style

df_prox.loc[df_prox.ID==381917.0]

df_pat.loc[df_pat.APPLN_ID=='15961384'].style

df_prox.loc[df_prox.C_LABEL=='CCS'].groupby(['ID', 'P_LABEL']).agg({'score': sum}).unstack(fill_value=0)

df_prox.loc[df_prox.C_LABEL=='CCS'].groupby(['ID', 'P_LABEL']).size().unstack(fill_value=0)

# Now create a dictionary which relates tags to technology classes.

df.reset_index(drop=True, inplace=True)

for tech_field in tech_fields:
    dummy_vec = []
    for ind, row in df.iterrows():
            dummy = tech_field in row.TAGS
            dummy_vec.append(dummy)
    df[tech_field] = pd.Series(dummy_vec)    

df.loc[df.air, ['ID', 'NAME', 'SHORT_DESCRIPTION', 'PRIMARY_TAG', 'air']]
