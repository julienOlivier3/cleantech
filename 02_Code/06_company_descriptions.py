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

# # Process company descriptions

# Read raw company descriptions
with open(r"Q:\Meine Bibliotheken\Research\Green_startups\02_Data\06_CompanyDescriptions\texte_taetigkeit.LST", "r") as f:
    temp = f.read().splitlines()

# Transfer into dataframe
df_desc = pd.DataFrame([i.split('\t') for i in temp], columns=['CREFO', 'DESC_de'])

# Clean columns
df_desc['CREFO'] = df_desc.CREFO.apply(lambda x: x.strip())
df_desc['DESC_de'] = df_desc.DESC_de.apply(lambda x: str.replace(x, 'Eingetragener Gegenstand:', '').strip()) # remove leading and trailing whitespaces and 'Eingetragener Gegenstand:'
df_desc['DESC_de'] = df_desc.DESC_de.apply(lambda x: re.sub('\s{2,}', ' ', x)) # remove leading and trailing whitespaces and 'Eingetragener Gegenstand:'

df_desc.DESC_de.sample(10).values

df_desc = df_desc.loc[df_desc.DESC_de.apply(lambda x: 'Kosmetik' in x)]


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

df_desc = pd.DataFrame(read_cache(here(r'02_Code/.pycache/company_desc_translations.pkl')), columns=['CREFO', 'DESC_en'])
