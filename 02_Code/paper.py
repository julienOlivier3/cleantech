# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# # Mapping technologies to business models: 
# ## An application to clean technologies and entrepreneurship
# Code supporting the findings in the paper

# +
import pandas as pd
import numpy as np
import pickle
from pyprojroot import here
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
model._first_module().max_seq_length = 510 # increase maximum sequence length which is 128 by default

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# # Patent corpus 

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Preparation 
# -

# Read patent data
df_pat = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))
df_pat.shape

df_pat.loc[df_pat.Y02==1].head(5)

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ## Patent count 
# -

# Drop patents without abstract
df_pat = df_pat.loc[df_pat.LEMMAS.notnull()]

print(' Number of patents: ', len(df_pat.APPLN_ID.drop_duplicates()), '\n',
      'Number of cleantech patents: ', len(df_pat.loc[df_pat.Y02==1].APPLN_ID.drop_duplicates()))

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Patent classes 
# -

# Typically, patents are labeled with more than just one technology class. This is particularly true for patents labeled with a cleantech class (Y02) as the Y02 taxonomy has been introduced as complementary scheme to the existing classification system.

# Create a mapping dictionary from CLEANTECH_MARKET to relevant CPCs and list with cleantech fields
df_map = pd.read_excel(here('./01_Data/02_Firms/df_cleantech_firms_label.xlsx'), sheet_name='Tag2CPC')
df_map = df_map.loc[df_map.ABBREVIATION.notnull() & df_map.CPC.notnull(),]
tech2market = dict(zip(df_map['CPC'], df_map['ABBREVIATION']))
cleantech_fields = list(set(tech2market.values()))
# remove ICT
cleantech_fields = [field for field in cleantech_fields if field != 'ICT']
cleantech_fields = ['Adaption', 'Battery', 'Biofuels', 'CCS', 'E-Efficiency', 'Generation', 'Grid', 'Materials', 'E-Mobility', 'Water']
print(cleantech_fields)

# Create dichotomous columns for each technology class (cleantech and non-cleantech).

techclasses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Adaption', 'Battery', 'Biofuels', 'CCS', 'E-Efficiency', 'Generation', 'Grid', 'Materials', 'E-Mobility', 'Water']

df_pat[techclasses] = 0

for techclass in techclasses:
    df_pat[techclass] = [techclass in row for row in df_pat.CLEANTECH_MARKET]
    df_pat[techclass] = df_pat[techclass].map({True: 1, False: 0})

# Add a column reflecting if the patent has been assigned to *any* non-cleantech class.

# + tags=[]
df_pat['non-cleantech'] = df_pat[set(techclasses).difference(cleantech_fields)].apply(sum, axis=1) > 0
df_pat['non-cleantech'] = df_pat['non-cleantech'].map({True: 1, False: 0})
# -

# Load similarity measures specifically designed for dichotomous variables (following the logic of F-scores).

from scipy.spatial.distance import dice, jaccard, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

# Correlation matrix.

cols = techclasses
corr_matrix = pd.DataFrame(index=cleantech_fields, columns=cols, dtype=np.float64)
corr_matrix

# +
cols = techclasses
corr_matrix = pd.DataFrame(index=cleantech_fields, columns=cols, dtype=np.float64)

for cleantech in cleantech_fields:
    corr_matrix.loc[cleantech] = (1 - df_pat.loc[df_pat[cleantech]==1, cols].corr(jaccard).loc[cleantech])
    
corr_matrix
# -

# Correlation as figure.

corr_matrix

# + tags=[]
sns.set_theme(style="white", font_scale=1.6)

# Compute the correlation matrix
#corr = corr_matrix[techclasses]
corr = corr_matrix[cleantech_fields]

# Generate a mask for the upper triangle
#mask = np.hstack(
#     (np.zeros_like(corr[set(techclasses).difference(cleantech_fields)], dtype=bool),
#     np.triu(np.ones_like(corr[cleantech_fields], dtype=bool)))
#)
mask = np.triu(np.ones_like(corr[cleantech_fields], dtype=bool))

# Name cleaning
corr.rename(index={'E-Mobility': 'Mobility'}, columns={'E-Mobility': 'Mobility'}, inplace=True)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 15))

# Generate a custom diverging colormap
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=1.8, reverse=False, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatm = sns.heatmap(corr,
                    mask=mask,
                    cmap=cmap,
                    #vmax=.3,
                    vmin=0,
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .4})

plot = heatm.set_yticklabels(heatm.get_yticklabels(), rotation=0)
plot = heatm.set_xticklabels(heatm.get_xticklabels(), rotation=45, ha="right")

# +
sns.set_theme(style="white", font_scale=1.6)

# Compute the correlation matrix
#corr = corr_matrix[techclasses]
corr = corr_matrix[techclasses[0:8]]

# Generate a mask for the upper triangle
#mask = np.hstack(
#     (np.zeros_like(corr[set(techclasses).difference(cleantech_fields)], dtype=bool),
#     np.triu(np.ones_like(corr[cleantech_fields], dtype=bool)))
#)

# Name cleaning
corr.rename(index={'E-Mobility': 'Mobility'}, columns={'E-Mobility': 'Mobility'}, inplace=True)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 15))

# Generate a custom diverging colormap
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=1.8, reverse=False, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatm = sns.heatmap(corr,
                    #mask=mask,
                    cmap=cmap,
                    #vmax=.3,
                    vmin=0,
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .4})

plot = heatm.set_yticklabels(heatm.get_yticklabels(), rotation=0)

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## Semantic technology descriptions
# -

# Use L-LDA to model semantic technology descriptions as probability distributions over vocabulary in patent corpus.

import tomotopy as tp

stoplist = ['and/or', '/h', 't1', 'dc', 'mm', 'wt', '113a', '115a', 'ofdm', 'lpwa']

# Instantiate labelled LDA model
model = tp.PLDAModel(tw=tp.TermWeight.IDF, topics_per_label=1, 
                     #latent_topics=8,
                     seed=333)

# Add documents to model
dropped_patents = []
for index, row in tqdm(df_pat.iterrows()):
    clean_document = row.LEMMAS
    # Remove some additional stopwords
    clean_document = [token for token in clean_document if token not in stoplist]
    # Remove Y04 and Y10 tag
    labels = [cpc for cpc in row.CLEANTECH_MARKET]
    # Add document and labels to model if clean_document is not empty
    if len(clean_document) > 0:
        model.add_doc(clean_document, labels=labels)
    else:
        dropped_patents.append(row.APPLN_ID)

print(f"{len(dropped_patents)} patents have been dropped. This is negligible!")

model.burn_in = 5
print('Start training model:')
for i in range(0, 100, 10):
    model.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

model.summary(topic_word_top_n=10)

model.topic_label_dict

n_relevant_words = 10000
df_topic_words = pd.DataFrame(columns = ['Topic', 'Word', 'Prob'])
for ind, topic in enumerate(model.topic_label_dict):
    temp = pd.DataFrame(model.get_topic_words(topic_id=ind, top_n=n_relevant_words), columns=['Word', 'Prob'])
    temp['Topic'] = topic
    df_topic_words = df_topic_words.append(temp)

df_topic_words

# Save to disk
df_topic_words.to_csv(here(r'.\03_Model\temp\df_topic_words_markets.txt'), sep='\t', encoding='utf-8')

# Create dataframe containing only words for which embeddings exist for all technology classes
df_excel = pd.DataFrame()
for topic in model.topic_label_dict:
    df_temp = df_topic_words.loc[df_topic_words.Topic==topic][:5000]
    df_temp = df_temp.drop(columns=['Topic']).rename(columns={'Word': topic+'_word', 'Prob': topic+'_prob'}).reset_index(drop=True)
    #print(topic, '\t', df_temp.shape)
    df_excel = pd.concat([df_excel, df_temp], axis=1)

df_excel

# Save to disk
df_excel.to_excel(here(r'.\03_Model\temp\df_topic_words_markets.xlsx'), encoding='utf-8')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Technology embeddings 
# -

df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words_markets.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
model._first_module().max_seq_length = 510 # increase maximum sequence length which is 128 by default

# Create technology embeddings using SBERT.

techclasses = set(df_topic_words.Topic.values)
n_words = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 75, 100, 200, 250, 300, 400, 500]
semantic_vectors_sbert = {}
for techclass in tqdm(techclasses):
    semantic_vectors_sbert[techclass] = {}
    for n_word in n_words:
        semantic_tech = ' '.join(str(i) for i in list(df_topic_words.loc[df_topic_words.Topic==techclass].head(n_word).Word.values))
        embedding = model.encode(semantic_tech)
        semantic_vectors_sbert[techclass][n_word] = embedding

# What is the dimension of the sequence embeddings?

print(f"SBERT translates word sequences into {len(semantic_vectors_sbert['Grid'][10])}-dimensional embeddings")

# Visualize the clean technology system.

from sklearn.manifold import TSNE


# Technology Embedding
def technology2embedding(df, model, technology, n_words):
    semantic_tech = ' '.join(str(word).lower() for word in list(df.loc[df.Topic==technology].head(n_words).Word.values))
    embedding = model.encode(semantic_tech)

    return(embedding)


def tsne(df, model, technology_labels, technology_lengths):
    "Creates and TSNE model and plots it"
    embs = []
    techs = []
    sizes = []

    for tech in technology_labels:
        for size in technology_lengths:
            sizes.append(size)
            techs.append(tech)
            embs.append(technology2embedding(df, model, tech, size))

    
    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2500, random_state=23)
    pca = tsne_model.fit_transform(embs)
    
    df = pd.DataFrame()
    df["Clean technology ($t$)"] = techs
    df["Number of words ($Q$)"] = sizes
    df["Component 1"] = pca[:,0]
    df["Component 2"] = pca[:,1]

    return(df)

df_tsne = tsne(df=df_topic_words,
               model=model,
               technology_labels=['Adaption', 'Battery', 'Biofuels', 'CCS', 'E-Efficiency', 'Generation', 'Grid', 'Materials', 'E-Mobility', 'Water'], 
               technology_lengths=[10,20,30,40,100])

df_tsne.loc[df_tsne.iloc[:,0]=='E-Mobility','Clean technology ($t$)'] = 'Mobility'

# +
sns.set(style="whitegrid") 

# Generate a custom diverging colormap
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=1.8, reverse=False, as_cmap=True)

#plt.subplots(figsize=(10, 15))

# Draw the heatmap with the mask and correct aspect ratio
sns.relplot(x = "Component 1", y = "Component 2", 
                hue = 'Clean technology ($t$)', 
                size = 'Number of words ($Q$)',
                alpha=0.9, 
                palette="BuGn",
                data=df_tsne, 
                sizes=(50, 700), 
                aspect=11.7/8.27)

# #g.set(xscale="log", yscale="log")
#g.ax.xaxis.grid(True, "minor", linewidth=.25)
#g.ax.yaxis.grid(True, "minor", linewidth=.25)
#g.despine(left=True, bottom=True)
plt.annotate('', xy=(30,-5), xytext=(+30,-15), textcoords='offset points',
             fontsize=16, arrowprops=dict(facecolor='red', connectionstyle='arc3,rad=-.2'))

plt.annotate('', xy=(-35,-12), xytext=(-30,+15), textcoords='offset points',
             fontsize=16, arrowprops=dict(facecolor='red', connectionstyle='arc3,rad=.2'))

plt.annotate('', xy=(2,7), xytext=(-30,+25), textcoords='offset points',
             fontsize=16, arrowprops=dict(facecolor='red', connectionstyle='arc3,rad=.15'))

plt.annotate('', xy=(28,30), xytext=(15,+35), textcoords='offset points',
             fontsize=16, arrowprops=dict(facecolor='red', connectionstyle='arc3,rad=-.05'))
# -
df_pat = df_pat.loc[df_pat.LEMMAS.apply(len)>0]

abstracts = df_pat.ABSTRACT.values
abstracts.shape

embeddings = model.encode(abstracts[:1000],  show_progress_bar=True)

embeddings.shape[0]/abstracts.shape[0]*100

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Proximity measure 


# + [markdown] tags=[]
# ## Preparation 
# -

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from util import string_to_lemma, TP, TN, FP, FN, ACC, PRECISION, RECALL, F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Load the scraped cleantech firms.

df_clean = pd.read_csv(here('01_Data/02_Firms/df_cleantech_firms.txt'), sep='\t')

# Drop duplicate firms (i.e. firms which made it to the Cleantech-100 list in subsequent years)
df_clean = df_clean.drop_duplicates('ID')
# Drop row with missings only
df_clean = df_clean.loc[df_clean.ID.notnull(),:]

# Read the mapping from tags to cleantech markets.

df_map = pd.read_excel(here('./01_Data/02_Firms/df_cleantech_firms_label.xlsx'), sheet_name='Tag2CPC')

df_map = df_map.loc[df_map.PRIMARY_TAG.notnull() & df_map.ABBREVIATION.notnull(), ['PRIMARY_TAG', 'ABBREVIATION']].drop_duplicates()
tag2market = dict(zip(df_map.PRIMARY_TAG, df_map.ABBREVIATION))
cleantech_fields = list(set(df_map.ABBREVIATION))
cleantech_fields

# Note that energy storage is mapped to E-Mobility and Grid. To solve this double mapping, I assign energy storage to E-Mobility if the company is additionally tagged with 'transportation & logistics'.

df_clean.loc[(df_clean.PRIMARY_TAG=='energy storage') & df_clean.TAGS.apply(lambda x: 'transportation & logistics' in x), 'PRIMARY_TAG'] = 'energy storage transportation'

# Generate cleantech market class
df_clean['CLEANTECH_MARKET'] = df_clean.PRIMARY_TAG.map(tag2market)

# Reduce to those companies for which cleantech market assignment worked (companies with 'other cleantech' tag are dropped)
df_clean = df_clean.loc[df_clean.CLEANTECH_MARKET.notnull()]

# How many cleantech firms are in the label?

print(f"There are {len(df_clean)} distinct cleantech firms in the sample of business summaries.")

# Load the scraped S&P 500 firms.

df_sp = pd.read_csv(here('01_Data/02_Firms/df_all_sp_firms.txt'), sep='\t')

# Drop companies without business description
df_sp = df_sp.loc[df_sp.BUSINESS_SUMMARY.notnull()]

# Check if there are cleantech firms listed on S&P 500
drop = set(df_sp.SYMBOL.values).intersection(set(df_clean.TICKER_SYMBOL.drop_duplicates().values))
drop

# Drop cleantech companies listed on S&P 500
df_sp = df_sp.loc[~df_sp.SYMBOL.isin(drop)]

# + active=""
# # Manual cleantech check
# drop = {'VLO', 'APD', 'CMI', 'BWA', 'ECL', 'WM', 'MPC', 'XYL', 'AWK', 'CF', 'PXD', 'CMS', 'DTE', 'ETN', 'IDXX', 'JCI', 'NEE', 'ED', 'DOV', 'AES', 'WEC', 'PNR', 'EIX', 'DUK', 'FMC', 'RSG', 'ADM'}
# # Drop cleantech companies listed on S&P 500
# df_sp = df_sp.loc[~df_sp.SYMBOL.isin(drop)]
# -

# How many cleantech firms are in the label?

print(f"There are {len(df_sp)} distinct S&P 500 firms in the sample of business summaries that are not on the Cleantech 100 list.")


# Text concatenator
def create_joint_string(x, columns = ['SHORT_DESCRIPTION', 'LONG_DESCRIPTION', 'PRODUCTS_DESCRIPTION', 'OVERVIEW']):
    return(' '.join([i for i in list(x[columns].values) if not pd.isnull(i)]))    


# Company Embeddings (all)
def allcompany2embedding(df1, df2, model, columns=['SHORT_DESCRIPTION'], stop_words = [], lemmatize=True):
    
    # Text concatenation
    df1['ID'] = df1.ID.astype(int).astype(str)
    df1['CLEANTECH'] = 1
    df1['CLEANTECH_MARKET'] = 'None'
    df1['DESC'] = df1.apply(lambda x: create_joint_string(x, columns=columns), axis=1)
    df1 = df1[['ID', 'CLEANTECH', 'CLEANTECH_MARKET', 'DESC']]
    
    sp500_cleantechs = {'VLO', 'APD', 'CMI', 'BWA', 'ECL', 'WM', 'MPC', 'XYL', 'AWK', 'CF', 'PXD', 'CMS', 'DTE', 'ETN', 'IDXX', 'JCI', 'NEE', 'ED', 'DOV', 'AES', 'WEC', 'PNR', 'EIX', 'DUK', 'FMC', 'RSG', 'ADM'}
    df2 = df2.loc[df2.BUSINESS_SUMMARY.notnull()].copy()
    df2['ID'] = df2.SYMBOL
    df2['CLEANTECH'] = df2['ID'].apply(lambda x: 1 if x in sp500_cleantechs else 0)
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


c, clean, c_tech, c_emb = allcompany2embedding(df_clean, df_sp, model, columns = ['SHORT_DESCRIPTION'], stop_words = [], lemmatize=False)

# + [markdown] tags=[]
# ## Property 1
# -

# It should allow to differentiate non-cleantech firms form cleantech firms, i.e. a company whose business model is unrelated to clean technologies should be distant from any of the clean technology embeddings.

# Embeddings and cosine similarity
tech_proxs = []
techs = cleantech_fields
n_words = list(np.arange(1, 31, 1))
n_words.extend(list(np.arange(40, 320, 20)))
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

# Drop ICT as it gives super high proximity to SP500
df_prox = df_prox.loc[df_prox.TECHNOLOGY!='ICT']

df_temp = df_prox.groupby(['COMPANY', 'N_WORDS']).agg({'TECHNOLOGY_PROXIMITY': max}).reset_index().merge(df_prox.groupby(['COMPANY', 'CLEANTECH', 'TECHNOLOGY', 'N_WORDS']).agg({'TECHNOLOGY_PROXIMITY': max}).reset_index())
df_temp

# +
fig, axes = plt.subplots(figsize=(9,5), ncols=1, nrows=1)


palette = {0: 'lightgrey', 1: 'green'}
ax = sns.boxplot(
    x="N_WORDS", y="TECHNOLOGY_PROXIMITY", hue="CLEANTECH",
    data=df_temp.loc[df_temp.N_WORDS.isin([i for i in n_words if i not in [21, 22, 23, 24, 25, 26, 27, 28, 29, 220, 240, 260, 280, 300]])], 
    linewidth=1, whis=(1, 99), fliersize=1.5, palette=palette, showfliers=True)
#ax.set_title('Proximity to ' + tech + ' (based on $N_{' + tech + '}$ = ' + str(N_true) + ', ' + '$N_{non-' + tech + '}$ = ' + str(N_false) + ')')
ax.set_xlabel('Number of words in technology descriptions ($Q$)')
ax.set_ylabel(r'Technological proximity')
ax.legend_.set_title('Label')
new_labels = ['non-cleantech', 'cleantech']
for t, l in zip(ax.legend_.texts, new_labels):
    t.set_text(l)
#plt.suptitle("Technological proximity based on labeled cleantech firms", size=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# -

# Do a dev-test split to conduct tuning of hyperparameters n_words and threshold and then evaluate performance on test set.

# + tags=[]
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
    for th in np.arange(0.1, 0.9, 0.01):
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
# Tune hyperparameters on the test set using grid search
class_metrics = []
n_word = 15
th = 0.27

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

print(classification_report(true, pred))

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Property 2 
# -

# It should position cleantech companies closest to their most relevant technologies, i.e. a company specialized in wind energy should be identified by a relatively high proximity to the technology embedding for renewable energy generation, while at the same time it should show little to no proximity to the embedding of e.g. water and wastewater technologies.

# Two objective statistics are reported for this purpose:
# 1. fraction of cleantech labeled firm within the top 1% of closest firms to a particular technology
# 2. Wilcoxon signed-rank test to show whether second closest proximity measure within the top 1% is significantly smaller than proximity to closest technology.

import scipy.stats as stats

# +
n_word = 15
top_percent = 0.01

df_search = pd.DataFrame()
df_temp = df_prox.loc[df_prox.N_WORDS==n_word]

for cleantech_field in cleantech_fields:
    if cleantech_field=='ICT':
        pass
    else:
        df_tech = df_temp.loc[df_temp.TECHNOLOGY==cleantech_field].sort_values('TECHNOLOGY_PROXIMITY', ascending=False)
        df_tech = df_tech.head(int(np.round(df_tech.shape[0]*top_percent)))
        y = []
        for company in df_tech.COMPANY.values:
            temp = df_temp.loc[(df_temp.COMPANY==company) & (df_temp.TECHNOLOGY!=cleantech_field), 'TECHNOLOGY_PROXIMITY'].max() 
            y.append(temp)
        df_tech['TECHNOLOGY_PROXIMITY_2ND'] = y
        df_search = df_search.append(df_tech[['TECHNOLOGY', 'COMPANY', 'CLEANTECH', 'TECHNOLOGY_PROXIMITY', 'TECHNOLOGY_PROXIMITY_2ND']])


# -

def aggregate_by_cleantech(x, th, alternative='greater'):
    d = {}
    d['FRACTION_CLEANTECH'] = x['CLEANTECH'].sum()/len(x)
    d['FRACTION_BELOW_CUTOFF'] = x['TECHNOLOGY_PROXIMITY_2ND'].apply(lambda i: i < th).sum()/len(x)
    d['WILCOXON'] = stats.wilcoxon(x['TECHNOLOGY_PROXIMITY'], x['TECHNOLOGY_PROXIMITY_2ND'], alternative=alternative)
    return pd.Series(d, index=['FRACTION_CLEANTECH', 'FRACTION_BELOW_CUTOFF', 'WILCOXON'])


df_search.groupby('TECHNOLOGY').apply(lambda x: aggregate_by_cleantech(x, th=0.27))

# Now show the top 1% CCS business summaries

df_CCS = df_temp.loc[df_temp.TECHNOLOGY=='CCS'].sort_values('TECHNOLOGY_PROXIMITY', ascending=False).head(10)
df_CCS.merge(df_clean[['ID', 'SHORT_DESCRIPTION']], left_on='COMPANY', right_on='ID')[['COMPANY', 'TECHNOLOGY_PROXIMITY', 'SHORT_DESCRIPTION']].style

# + [markdown] tags=[]
# # Clean technologies and entrepreneurship 
# -

# Load embedded texts.

df_startup = pd.read_csv(here("01_Data/02_Firms/03_StartupPanel/df_gp_impute.txt"), sep="\t")
df_startup.rename(columns={'E_Mobility': 'E-Mobility', 'E_Efficiency': 'E-Efficiency'}, inplace=True)
df_startup.shape

# + [markdown] tags=[]
# ## Descriptives company descriptions
# -

print(f"There are {len(df_startup)} distinct start-ups in the survey for which business summaries exist.")

from util import nlp
tqdm.pandas()

df_startup['TOKEN_LEN'] = df_startup.text_en.progress_apply(lambda x: len(nlp(x)))
df_startup['DIGIT_LEN'] = df_startup.text_en.progress_apply(lambda x: len(x))
df_startup['TOKENS'] = df_startup.text_en.progress_apply(lambda x: [token for token in nlp(x)])

df_startup.TOKEN_LEN.describe()

# Vocabulary size
df_startup = df_startup.reset_index(drop=True)
vocab_list = []
for i in range(df_startup.shape[0]):
    temp = df_startup.iloc[i].TOKENS
    vocab_list.extend(temp)
len(set(vocab_list))

# ## Cleantech entrants

cleantech_fields = ['Adaption', 'Battery', 'Biofuels', 'CCS', 'E-Efficiency', 'Grid', 'Generation', 'Materials', 'E-Mobility' , 'Water']

df_prox = pd.melt(df_startup[cleantech_fields], var_name="Clean technology field", value_name="Technological proximity")
df_prox.loc[df_prox["Technological proximity"].notnull()]

df_prox.loc[df_prox.iloc[:,0]=='E-Mobility','Clean technology field'] = 'Mobility'

# +
plt.rcParams['text.usetex'] = True
plt.figure(figsize = (9, 5))
sns.set(style="whitegrid")
palette = sns.cubehelix_palette(n_colors=10, start=2, rot=0, dark=0.35, light=0.9, reverse=False, as_cmap=False)

#ax = sns.swarmplot(x="Clean technology field", y="Technological proximity", data=df_prox, palette=pal,
#                   dodge=True, alpha=0.5)
ax = sns.boxplot(x="Clean technology field", y="Technological proximity", data=df_prox, color='lightgray',
                linewidth=1, 
                #whis=(1, 99), 
                palette=palette, 
                showfliers=True, 
                boxprops=dict(alpha=.4), 
                flierprops=dict(markerfacecolor="r", markersize=5, linestyle='none', markeredgecolor='r', alpha=0.8))
ax.axhline(y=0.27, xmin=0, xmax=0.03, linestyle='--', linewidth=1, color='r')
ax.axhline(y=0.27, xmin=0.18, linestyle='--', linewidth=1, color='r')
ax.text(x=-0.1, y=0.265, s=r'\textsc{TechProx}$_{min}$', fontsize=10)
plt.show()
# -

df_startup[['gpkey', 'tech_prox', 'tech', 'text_en']].sort_values('tech_prox', ascending=False).head(20).style

df_startup.tech.value_counts(dropna=False)

print(f"There are {len(df_startup.loc[df_startup.tech_prox>0.27])} distinct start-ups in the survey classified as cleantech.")
