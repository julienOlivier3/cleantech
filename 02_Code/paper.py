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

# + [markdown] tags=[]
# # "Mapping technologies to business models: An application to clean technologies and entrepreneurship"
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
# -

# # Patent corpus 

# Read patent data
df_pat = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))
df_pat.shape

df_pat.loc[df_pat.Y02==1].head(5)

# ## Patent count 

# Drop patents without abstract
df_pat = df_pat.loc[df_pat.LEMMAS.notnull()]

print(' Number of patents: ', len(df_pat.APPLN_ID.drop_duplicates()), '\n',
      'Number of cleantech patents: ', len(df_pat.loc[df_pat.Y02==1].APPLN_ID.drop_duplicates()))

# ## Patent classes 

# Typically, patents are labeled with more than just one technology class. This is particularly true for patents labeled with a cleantech class (Y02) as the Y02 taxonomy has been introduced as complementary scheme to the existing classification system.

# Create a mapping dictionary from CLEANTECH_MARKET to relevant CPCs and list with cleantech fields
df_map = pd.read_excel(here('./01_Data/02_Firms/df_cleantech_firms_label.xlsx'), sheet_name='Tag2CPC')
df_map = df_map.loc[df_map.ABBREVIATION.notnull() & df_map.CPC.notnull(),]
tech2market = dict(zip(df_map['CPC'], df_map['ABBREVIATION']))
cleantech_fields = list(set(tech2market.values()))
print(cleantech_fields)

# Create dichotomous columns for each technology class (cleantech and non-cleantech).

techclasses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'E-Mobility', 'Battery', 'Adaption', 'Materials', 'Generation', 'ICT', 'Water', 'E-Efficiency', 'Biofuels', 'CCS', 'Grid']

df_pat[techclasses] = 0

for techclass in techclasses:
    df_pat[techclass] = [techclass in row for row in df_pat.CLEANTECH_MARKET]
    df_pat[techclass] = df_pat[techclass].map({True: 1, False: 0})

# Add a column reflecting if the patent has been assigned to *any* non-cleantech class.

df_pat['non-cleantech'] = df_pat[set(techclasses).difference(cleantech_fields)].apply(sum, axis=1) > 0
df_pat['non-cleantech'] = df_pat['non-cleantech'].map({True: 1, False: 0})

# Load similarity measures specifically designed for dichotomous variables (following the logic of F-scores).

from scipy.spatial.distance import dice, jaccard, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

# Correlation matrix.

# +
cols = ['non-cleantech'] + techclasses
corr_matrix = pd.DataFrame(index=cleantech_fields, columns=cols, dtype=np.float64)

for cleantech in cleantech_fields:
    corr_matrix.loc[cleantech] = (1 - df_pat.loc[df_pat[cleantech]==1, cols].corr(jaccard).loc[cleantech])
    
corr_matrix
# -

# Correlation as figure.

# + tags=[]
sns.set_theme(style="white")

# Compute the correlation matrix
# corr = corr_matrix[['non-cleantech'] + cleantech_fields]
corr = corr_matrix[cleantech_fields]

# Generate a mask for the upper triangle
# mask = np.hstack(
#     (np.zeros_like(corr[['non-cleantech']], dtype=bool),
#     np.triu(np.ones_like(corr[cleantech_fields], dtype=bool)))
# )
mask = np.triu(np.ones_like(corr[cleantech_fields], dtype=bool))

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
                    cbar_kws={"shrink": .5})

plot = heatm.set_yticklabels(heatm.get_yticklabels(), rotation=0)
# -

# ## Semantic technology descriptions

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

# +
import pandas as pd
import numpy as np
import gensim
import nltk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import dice, jaccard, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule
from tqdm import tqdm

from step0_setup import engine, executeSQL
# -

engine.table_names()

# %%time
df_pat = executeSQL(f"""SELECT * FROM PATENTS""")

df_pat.shape

# Colors

sns.color_palette("deep")

sns.color_palette("deep")[2]

green = matplotlib.colors.to_hex(list(sns.color_palette("deep")[2]))

# + [markdown] tags=[] heading_collapsed="true"
# # Number of patents and companies 
# -

# Note there are duplicated APPLN_IDs if patent has been filed jointly by more than one firm.

df_pat[df_pat.APPLN_ID.duplicated(keep=False)]

# Note, however, that there are some patents which have been filed multiple times? Not considered yet...

df_pat[df_pat.DOCDB_FAMILY_ID.duplicated(
    keep=False)][~df_pat.APPLN_ID.duplicated(
        keep=False)].sort_values('DOCDB_FAMILY_ID')

print(' Number of Y02 patents: ', len(np.unique(df_pat.APPLN_ID)), '\n',
      'Number of companies: ', len(np.unique(df_pat.CREFO)))


# + [markdown] tags=[]
# # Patent analysis 

# + [markdown] tags=[] heading_collapsed="true"
# ## Type of patent applicant 
# -

df_pat.PSN_SECTOR.value_counts(dropna=False)

# + [markdown] tags=[] heading_collapsed="true"
# ## Y02 distribution 
# -

# Create df with unique patents only.

df_pat_unique = df_pat.loc[~df_pat.APPLN_ID.duplicated(keep='first'), [
    'APPLN_ID', 'EARLIEST_FILING_YEAR', 'GRANTED', 'DOCDB_FAMILY_ID',
    'Y02_CLASSES', 'N_APPLICANTS', 'COLLAB_TYPE'
]]

df_pat_unique.head(3)

# Create list including every Y02 class appearing.

lst_y02 = df_pat_unique.Y02_CLASSES.apply(lambda x: x.split("; ")).to_list()
lst_y02 = [item for sublist in lst_y02 for item in sublist]

# Count how often Y02 class appears.

df_y02 = pd.concat([
    pd.DataFrame(lst_y02).value_counts(normalize=False),
    pd.DataFrame(lst_y02).value_counts(normalize=True)
],
                   axis=1)
df_y02.columns = ['Absolute', 'Relative']
df_y02

# + [markdown] tags=[] heading_collapsed="true"
# ## Y02 correlation 
# -

# Y02E and Y02P are often jointly tackled in one and the same patent.

df_pat_unique.Y02_CLASSES.value_counts().head(10)

# Create dichotomous columns for each Y02 class.

Y02classes = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']

df_pat_unique[Y02classes] = 0

for j in Y02classes:
    df_pat_unique[j] = [j in i for i in df_pat_unique.Y02_CLASSES]
    df_pat_unique[j] = df_pat_unique[j].map({True: 1, False: 0})

# Load similarity measures specifically designed for dichotomous variables (following the logic of F-scores).

from scipy.spatial.distance import dice, jaccard, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

# Sanity check whether dice (or related) measure give expected results.

1 - df_pat_unique.loc[(df_pat_unique.Y02E == 1) &
                      (df_pat_unique.Y02P == 1)][Y02classes].corr(dice)

# Yes they do!

# Correlation matrix.

1 - df_pat_unique[Y02classes].corr(dice)

# Correlation as figure.

# + tags=[]
sns.set_theme(style="white")

# Compute the correlation matrix
corr = 1 - df_pat_unique[Y02classes].corr(dice)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 7))

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
                    cbar_kws={"shrink": .5})

plot = heatm.set_yticklabels(heatm.get_yticklabels(), rotation=0)

# + [markdown] tags=[] heading_collapsed="true"
# ## Patent corpus 
# -

# %time
df_abs = executeSQL(f"""SELECT * FROM ABSTRACTS""")

df_abs.shape

# There are more unique APPLN_ID in df_pat than df_abs. Check number of joint APPLN_ID.

df_abs=df_abs.merge(df_pat_unique[['APPLN_ID', 'Y02_CLASSES']], how='left', left_on='APPLN_ID', right_on='APPLN_ID')

df_abs.Y02_CLASSES.isnull().sum()

# For all 40457 of the patents exist abstracts.

# Distribution of number of words.

df_abs.ABSTRACT.apply(lambda x: len(x.split())).sort_values()

df_abs.iloc[5353, 1]

df_abs.iloc[5353, 0]

plt.figure()
ax = df_abs.loc[~(df_abs.APPLN_ID==253881),:].ABSTRACT.apply(lambda x: len(x.split())).plot(kind='hist', bins=50, color=green)
#plt.yscale('log', nonpositive='clip')
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.show()

# Average number of words.

df_abs.loc[~(df_abs.APPLN_ID==253881),:].ABSTRACT.apply(lambda x: len(x.split())).describe()

# Descriptive statistics on sentence level.

df_abs.loc[~(df_abs.APPLN_ID==253881),:].ABSTRACT.apply(lambda x: len(nltk.tokenize.sent_tokenize(x))).describe()

# Load dictionary that resulted afte some cleaning (remove punctuation, lowercasing, remove stopwords, remove digits).

dictionary_pat = gensim.utils.SaveLoad.load(
    r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\dictionary_abstracts.dict'
)

# Number of words in dictionary

len(dictionary_pat)

print(dictionary_pat)

with open(
        r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\corpus_abstracts.pkl',
        'rb') as f:
    corpus = pickle.load(f)

# Count frequencies of words occuring in the corpus.

# +
import itertools
from collections import defaultdict

total_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_count[dictionary_pat[word_id]] += word_count
    
total_count = list(total_count.items())
# -

# Extend list of stopwords or remove tokens from dictionary _ex post_ via filter_tokens().

token2remove = ['with', 'which', 'one', 'or', 'by', 'from', 'that', 'be', 'as', 'has', 'can', 'first', 'second', 'into', 'between', 'having', 'wherein',
               'such', 'least', 'comprises', 'said']

# Drop words from dictionary.

# + tags=[]
# Top words
sorted([x for x in total_count if x[0] not in token2remove], key=lambda x: x[1], reverse=True)[0:20]
# -

dictionary_pat.filter_tokens(bad_ids=[dictionary_pat.token2id[token] for token in token2remove])

len(dictionary_pat)



# + [markdown] tags=[]
# # Firm analysis 
# -

df_pat

df_pat['Y02_CLASSES'] = df_pat.Y02_CLASSES.apply(lambda x: str(x).split('; '))

df_pat_grouped = df_pat.groupby('CREFO')
pat_counts = df_pat_grouped.size().to_frame(name='PATENT_COUNT')
df_pat_firm = pat_counts.join(
    df_pat_grouped.agg({
        'PSN_NAME': lambda x: list(set(x)),
        'PSN_SECTOR': lambda x: list(set(x)),
        'Y02_CLASSES': lambda x: sorted(list(set([item for sublist in x for item in sublist]))),
    })).reset_index()

# Add information concerning sucessful Y02 applications only.

df_pat_grouped_granted = df_pat.loc[df_pat.GRANTED=='Y',:].groupby('CREFO')
pat_counts_granted = df_pat_grouped_granted.size().to_frame(name='PATENT_COUNT')
df_pat_firm_granted = pat_counts_granted.join(
    df_pat_grouped_granted.agg({
        'Y02_CLASSES': lambda x: sorted(list(set([item for sublist in x for item in sublist]))),
    })).rename({'Y02_CLASSES': 'Y02_CLASSES_GRANTED',
               'PATENT_COUNT': 'PATENT_COUNT_GRANTED',}, axis=1).reset_index()

# Merge info concerning all patents and granted patents only
df_pat_firm = df_pat_firm.merge(df_pat_firm_granted, how='left', left_on='CREFO', right_on='CREFO')

# If patent count for granted patents is missing for a firm then impute with 0
df_pat_firm['PATENT_COUNT_GRANTED'] = df_pat_firm.PATENT_COUNT_GRANTED.apply(lambda x: np.where(pd.isna(x), 0, x)).astype(int)

# Order columns
df_pat_firm = df_pat_firm[['CREFO', 'PSN_NAME', 'PSN_SECTOR', 'PATENT_COUNT', 'PATENT_COUNT_GRANTED', 'Y02_CLASSES', 'Y02_CLASSES_GRANTED']]

# + [markdown] tags=[]
# ## Patent count per firm 
# -

df_pat_firm.sort_values('PATENT_COUNT_GRANTED', ascending=False)

# Plot distribution of number of succesfully filed patents per firm.

plt.figure()
ax = df_pat_firm.loc[df_pat_firm.PATENT_COUNT_GRANTED>0,:].PATENT_COUNT_GRANTED.hist(bins=40, edgecolor='white', color=green)
plt.yscale('log', nonpositive='clip')
plt.xlabel("Number of Y02 patents")
plt.ylabel("Frequency")
plt.show()

# Exclude outlier Siemens.

plt.figure()
ax = df_pat_firm.loc[(df_pat_firm.PATENT_COUNT_GRANTED>0) & ~(df_pat_firm.CREFO==3430000457),:].PATENT_COUNT_GRANTED.hist(bins=40, edgecolor='white', color=green)
plt.yscale('log', nonpositive='clip')
plt.xlabel("Number of Y02 patents")
plt.ylabel("Frequency")
plt.show()

# Look at relation between firm size and number of patents as well as number of technology fields.

# %%time
df_crefo = executeSQL(f"""SELECT * FROM CREFOS""")

df_crefo.head(3)

# Merge firm size to firm DF
df_pat_firm = df_pat_firm.merge(df_crefo[['CREFO', 'anzma', 'umsatz', 'age']], how='left', left_on='CREFO', right_on='CREFO')

# Create count of tech clusters
df_pat_firm['CLUSTER_COUNT_GRANTED'] = df_pat_firm['Y02_CLASSES_GRANTED'].apply(lambda x: 0 if type(x) == float else len(x))

df_pat_firm.head(3)

plt.figure()
sns.scatterplot(
    data=df_pat_firm.loc[(df_pat_firm.PATENT_COUNT_GRANTED>0)], x="umsatz", y="PATENT_COUNT_GRANTED", hue="CLUSTER_COUNT_GRANTED", size="CLUSTER_COUNT_GRANTED",
    sizes=(20, 200), hue_norm=(0, 7), legend="full", alpha=1, 
    palette=sns.cubehelix_palette(start=2, rot=0, dark=0.35, light=1, reverse=False, as_cmap=True)
)
plt.xscale('log', nonpositive='clip')
plt.xlabel("Turnover")
plt.ylabel("Number of Y02 patents")
legend = plt.legend(title="Number of \nclean-tech \nclusters", ncol=1)
plt.show()

# + [markdown] tags=[]
# ## Y02 correlation 
# -

Y02classes = ['Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W']

df_pat_firm[Y02classes] = 0

for j in Y02classes:
    df_pat_firm[j] = [j in i for i in df_pat_firm.Y02_CLASSES]
    df_pat_firm[j] = df_pat_firm[j].map({True: 1, False: 0})

# Correlation matrix.

1 - df_pat_firm[Y02classes].corr(dice)

# Correlation as figure.

# + tags=[]
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Compute the correlation matrix
corr = 1 - df_pat_firm[Y02classes].corr(dice)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 7))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(3, 100, as_cmap=True)
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
                    cbar_kws={"shrink": .5})

plot = heatm.set_yticklabels(heatm.get_yticklabels(), rotation=0)

# + [markdown] tags=[]
# ## Webtext corpus 
# -

df_pat_firm.head(3)

# %time
df_web = executeSQL(f"""SELECT * FROM TEXTS""")

df_web.shape

df_web.head(3)

# Distribution of number of words.

df_web.text.apply(lambda x: len(x.split())).sort_values()

df_web.iloc[229,0]

plt.figure()
ax = df_web.loc[~(df_web.CREFO==5190398809),:].text.apply(lambda x: len(x.split())).plot(kind='hist', bins=50, color=green) # Drop outlier 5190398809
#plt.yscale('log', nonpositive='clip')
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.show()

# Average number of words.

df_web.loc[~(df_web.CREFO==5190398809),:].text.apply(lambda x: len(x.split())).describe()

# Descriptive statistics on sentence level.

df_web.loc[~(df_web.CREFO==5190398809),:].text.apply(lambda x: len(nltk.tokenize.sent_tokenize(x))).describe()

# Load dictionary that resulted afte some cleaning (remove punctuation, lowercasing, remove stopwords, remove digits).

dictionary_web = gensim.utils.SaveLoad.load(
    r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\dictionary_webtexts.dict'
)

# Number of words in dictionary

len(dictionary_web)

print(dictionary_web)

with open(
        r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\corpus_webtexts.pkl',
        'rb') as f:
    corpus = pickle.load(f)

# Count frequencies of words occuring in the corpus.

# +
import itertools
from collections import defaultdict

total_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_count[dictionary_web[word_id]] += word_count
    
total_count = list(total_count.items())
# -

# Extend list of stopwords or remove tokens from dictionary _ex post_ via filter_tokens().

token2remove = ['google', 'en', 'la', 'cookie', 'et', 'press', 'search', 'go', 'gdpr', 'home', 'm', 'x', 'ag', 'overview', 'newsletter', 'career', 'tel', 'phone', 'center',
               'menu', 'dr', 'settings', 'youtube', 'des', 'events', 'english', 'facebook', 'navigation', 'y']

# Drop words from dictionary.

# + tags=[]
# Top words
sorted([x for x in total_count if x[0] not in token2remove], key=lambda x: x[1], reverse=True)[0:20]
# -

dictionary_web.filter_tokens(bad_ids=[dictionary_web.token2id[token] for token in token2remove])

len(dictionary_web)

# Merge both dictionaries.

dictionary_pat.merge_with(dictionary_web)

len(dictionary_pat)

# + [markdown] tags=[]
# # Model analysis 
# -

# Load dictionary, corpus and model

dictionary = gensim.corpora.Dictionary.load(r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\dictionary.dict')
corpus = pickle.load(open(r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\corpus_counts.pkl', 'rb'))
lda_model = gensim.models.ldamodel.LdaModel.load(r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\lda_100_counts.model')

# Clean corpus

corpus = [texts[1] for texts in tqdm(corpus)]

import pyLDAvis.gensim

# + jupyter={"outputs_hidden": true} tags=[]
lda_visualization = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary, sort_topics=False)
# -

pyLDAvis.display(lda_visualization)

pyLDAvis.save_html(lda_visualization, r'Q:\Meine Bibliotheken\Research\Green_startups\05_Model\04_LDA\pyldavis_100.html')
