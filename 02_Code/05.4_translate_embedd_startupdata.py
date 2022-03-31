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
from pyprojroot import here
from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()

# + [markdown] tags=[]
# # StratGreen 

# + [markdown] tags=[]
# ## Translation 
# -

df_startup = pd.read_csv(here('01_Data/02_Firms/df_startup_firms.txt'), sep='\t', encoding='utf-8')

df_startup.iloc[[1]].style

df_startup.LONG_DESCRIPTION.iloc[1]

GoogleTranslator().translate(df_startup.LONG_DESCRIPTION.iloc[1])

df_startup.loc[df_startup.LONG_DESCRIPTION.notnull()].LONG_DESCRIPTION.apply(len).plot(kind='hist')

df_temp = df_startup.loc[df_startup.LONG_DESCRIPTION.notnull()].head(3)

df_temp['LONG_DESCRIPTION_en'] = df_temp.LONG_DESCRIPTION.progress_apply(lambda x: GoogleTranslator().translate(x))

df_temp[['LONG_DESCRIPTION', 'LONG_DESCRIPTION_en']].style

df_temp = df_startup.loc[df_startup.LONG_DESCRIPTION.notnull()]

df_temp.shape

df_temp['LONG_DESCRIPTION_en'] = df_temp.LONG_DESCRIPTION.progress_apply(lambda x: GoogleTranslator().translate(x))

df_temp[['LONG_DESCRIPTION', 'LONG_DESCRIPTION_en']].tail(5)

df_startup = df_startup.merge(df_temp[['HREF', 'LONG_DESCRIPTION_en']], on='HREF', how='left')

df_startup.to_csv(here('01_Data/02_Firms/df_startup_firms_en.txt'), sep='\t', encoding='utf-8', index=False)

# ## Technological Proximity

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
model._first_module().max_seq_length = 510 # increase maximum sequence length which is 128 by default

cleantech_fields = ['Biofuels', 'Battery', 'CCS', 'Water', 'Adaption', 'E-Efficiency', 'Materials', 'E-Mobility', 'Grid', 'Generation']

# First create startup and technology embeddings.

print(f"The data comprises {df_startup.shape[0]} startups.")

# Drop samples without company description.

df_startup = df_startup.loc[df_startup.LONG_DESCRIPTION_en.notnull()]
df_startup.reset_index(inplace=True, drop=True)

print(f"For {df_startup.shape[0]} startups a company description exists.")


# Company Embeddings (all)
def company2embedding(df, model, stop_words = [], lemmatize=True):
       
    company = df.index.values
    
    # Lemmatization, stop word removal and sentence embedding
    if lemmatize:
        # Conduct lemmatization
        df['LONG_DESCRIPTION_en'] = df.DESC.apply(lambda x: string_to_lemma(x))
        # Remove stop words
        df['LONG_DESCRIPTION_en'] = df.DESC.apply(lambda x: [word for word in x if word not in stop_words])
        # Concatenate list of words to whitespace seperated string
        df['LONG_DESCRIPTION_en'] = df.DESC.apply(lambda x: ' '.join(str(i) for i in x))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.LONG_DESCRIPTION_en.values, show_progress_bar=True)
    
    # Stop word removal Sentence embedding
    else:
        # Remove stop words
        #df['DESC'] = df.DESC.apply(lambda x: ' '.join(word.lower() for word in x.split() if word.lower() not in stop_words))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.LONG_DESCRIPTION_en.values, show_progress_bar=True)

    return(company, embeddings)


c, c_emb = company2embedding(df_startup, model=model, stop_words=[], lemmatize=False)

# Read topic-proba-df
df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words_markets.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')


# Technology Embedding
def technology2embedding(df, model, technology, n_words):
    semantic_tech = ' '.join(str(word).lower() for word in list(df.loc[df.Topic==technology].head(n_words).Word.values))
    embedding = model.encode(semantic_tech).reshape(1, -1)

    return(embedding)


# Now calculate proximity.

# Embeddings and cosine similarity
tech_proxs = []
techs = cleantech_fields
n_word = 15
for tech in tqdm(techs):
    t_emb = technology2embedding(df_topic_words, model, technology=tech, n_words=n_word)
    tech_prox = cosine_similarity(c_emb, t_emb).reshape(len(c_emb),)
    tech_prox[tech_prox < 0] = 0
    tech_proxs.append(pd.DataFrame({'COMPANY': c, 'TECHNOLOGY': tech, 'TECHNOLOGY_PROXIMITY': tech_prox}))
    df_prox = pd.concat(tech_proxs)

df_prox.sort_values('TECHNOLOGY_PROXIMITY', ascending=False).head(20)

df_startup = df_startup.merge(df_prox.pivot(index='COMPANY', columns='TECHNOLOGY', values='TECHNOLOGY_PROXIMITY'), left_index=True, right_index=True)

df_temp = df_startup[['NAME', 'LONG_DESCRIPTION_en', 'Biofuels', 'Battery', 'CCS', 'Water', 'Adaption', 'E-Efficiency', 'Materials', 'E-Mobility', 'Grid', 'Generation']]

df_temp.sort_values('Generation', ascending=False).head(5).style

df_startup.to_csv(here('01_Data/02_Firms/df_startup_firms_en_prox.txt'), sep='\t', index=False)

# + [markdown] tags=[]
# # Start-up panel 

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Translation 
# -

df_startup = pd.read_csv(here('01_Data/02_Firms/df_gp.txt'), sep='\t', encoding='utf-8')

df_temp = df_startup.loc[df_startup.text.notnull()].head(3)

df_temp['text_en'] = df_temp.text.progress_apply(lambda x: GoogleTranslator().translate(x))

df_temp[['text', 'text_en']].style

df_temp = df_startup.loc[df_startup.text.notnull()]

df_temp.shape

df_temp['text_en'] = df_temp.text.progress_apply(lambda x: GoogleTranslator().translate(x))

df_temp[['text', 'text_en']].tail(5)

df_startup = df_startup.merge(df_temp[['crefo', 'text_en']], on='crefo', how='left')

df_startup.to_csv(here('01_Data/02_Firms/df_gp_en.txt'), sep='\t', encoding='utf-8', index=False)

# ## Technological Proximity

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
model._first_module().max_seq_length = 510 # increase maximum sequence length which is 128 by default

cleantech_fields = ['Biofuels', 'Battery', 'CCS', 'Water', 'Adaption', 'E-Efficiency', 'Materials', 'E-Mobility', 'Grid', 'Generation']

# First create startup and technology embeddings.

print(f"The data comprises {df_startup.shape[0]} startups.")

# Drop samples without company description.

df_startup = df_startup.loc[df_startup.text_en.notnull()]

print(f"For {df_startup.shape[0]} startups a company description exists.")


# Company Embeddings
def company2embedding(df, model, stop_words = [], lemmatize=True):
       
    company = df.index.values
    
    # Lemmatization, stop word removal and sentence embedding
    if lemmatize:
        # Conduct lemmatization
        df['text_en'] = df.DESC.apply(lambda x: string_to_lemma(x))
        # Remove stop words
        df['text_en'] = df.DESC.apply(lambda x: [word for word in x if word not in stop_words])
        # Concatenate list of words to whitespace seperated string
        df['text_en'] = df.DESC.apply(lambda x: ' '.join(str(i) for i in x))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.text_en.values, show_progress_bar=True)
    
    # Stop word removal Sentence embedding
    else:
        # Remove stop words
        #df['DESC'] = df.DESC.apply(lambda x: ' '.join(word.lower() for word in x.split() if word.lower() not in stop_words))
        # Create numpy array of sentence embedding
        embeddings = model.encode(df.text_en.values, show_progress_bar=True)

    return(company, embeddings)


c, c_emb = company2embedding(df_startup, model=model, stop_words=[], lemmatize=False)

# Read topic-proba-df
df_topic_words = pd.read_csv(here(r'.\03_Model\temp\df_topic_words_markets.txt'), sep='\t', encoding='utf-8', index_col='Unnamed: 0')


# Technology Embedding
def technology2embedding(df, model, technology, n_words):
    semantic_tech = ' '.join(str(word).lower() for word in list(df.loc[df.Topic==technology].head(n_words).Word.values))
    embedding = model.encode(semantic_tech).reshape(1, -1)

    return(embedding)


# Now calculate proximity.

# Embeddings and cosine similarity
tech_proxs = []
techs = cleantech_fields
n_word = 15
for tech in tqdm(techs):
    t_emb = technology2embedding(df_topic_words, model, technology=tech, n_words=n_word)
    tech_prox = cosine_similarity(c_emb, t_emb).reshape(len(c_emb),)
    tech_prox[tech_prox < 0] = 0
    tech_proxs.append(pd.DataFrame({'COMPANY': c, 'TECHNOLOGY': tech, 'TECHNOLOGY_PROXIMITY': tech_prox}))
    df_prox = pd.concat(tech_proxs)

df_startup = df_startup.merge(df_prox.pivot(index='COMPANY', columns='TECHNOLOGY', values='TECHNOLOGY_PROXIMITY'), left_index=True, right_index=True)

df_temp = df_startup[['crefo', 'text_en', 'Biofuels', 'Battery', 'CCS', 'Water', 'Adaption', 'E-Efficiency', 'Materials', 'E-Mobility', 'Grid', 'Generation']]

df_temp.sort_values('Generation', ascending=False).head(5).style

df_startup.to_csv(here('01_Data/02_Firms/df_gp_en_prox.txt'), sep='\t', index=False)
