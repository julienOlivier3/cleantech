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
import re
from tqdm import tqdm
import pickle as pkl
from pyprojroot import here
from util import translation_handler
import config

# # Process company descriptions

# Read raw company descriptions
with open(config.PATH_TO_COMPANY_DESCRIPTIONS + r"\texte_taetigkeit.LST", "r") as f:
    temp = f.read().splitlines()

# Transfer into dataframe
df_desc = pd.DataFrame([i.split('\t') for i in temp], columns=['CREFO', 'DESC_de'])

# Clean columns
df_desc['CREFO'] = df_desc.CREFO.apply(lambda x: x.strip())
df_desc['DESC_de'] = df_desc.DESC_de.apply(lambda x: str.replace(x, 'Eingetragener Gegenstand:', '').strip()) # remove leading and trailing whitespaces and 'Eingetragener Gegenstand:'
df_desc['DESC_de'] = df_desc.DESC_de.apply(lambda x: re.sub('\s{2,}', ' ', x)) # remove leading and trailing whitespaces and 'Eingetragener Gegenstand:'

df_desc.DESC_de.sample(10).values


# +
# Create a sample of non-tech firms
#df_desc = df_desc.loc[df_desc.DESC_de.apply(lambda x: 'Kosmetik' in x)]
# -

# ## Translation 

def text_translation(df):
    temp = list()
    for i in range(len(df)):
        temp.append((df.iloc[i]['CREFO'], translation_handler(df.iloc[i]['DESC_de'])))
    return(temp)


n = len(df_desc)
chunk_size = 100
for start in tqdm(range(0, n, chunk_size)):
    df_subset = df_desc.iloc[start:(start+chunk_size)]
    cdesc_translations = text_translation(df_subset)
    with open(here('02_Code/.pycache/company_desc_translations.pkl'), 'ab+') as f:
        f.write(pkl.dumps(cdesc_translations))

from util import read_cache

df_desc_en1 = pd.DataFrame(read_cache(here(r'02_Code/.pycache/company_desc_translations1.pkl')), columns=['CREFO', 'DESC_en'])
df_desc_en2 = pd.DataFrame(read_cache(here(r'02_Code/.pycache/company_desc_translations2.pkl')), columns=['CREFO', 'DESC_en'])

df_desc_en = pd.concat([df_desc_en1, df_desc_en2])

df_desc_en.shape, df_desc.shape

df_desc = df_desc.merge(df_desc_en, how="left", on="CREFO")

# Write to disk
df_desc.to_csv(here(r'01_Data/02_Firms/02_Company_Descriptions/df_desc.txt'), sep='\t', encoding='utf-8', index=False)
