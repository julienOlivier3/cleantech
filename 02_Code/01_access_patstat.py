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

# # Connect to PATSTAT and Extract Patent Data of German Companies

# # Setup

import config # configuration files includes API keys and paths
import pandas as pd
import numpy as np
from sqlalchemy import types, create_engine
import cx_Oracle
from pyprojroot import here
from tqdm import tqdm
import pickle as pkl

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Extract Data 
# -

# The following functions connect to the PATSTAT database.

cx_Oracle.init_oracle_client(lib_dir=config.PATH_TO_ORACLE_CLIENT) # path to Oracle instantclient (if not in PATH)
patstat_connection_string = 'oracle+cx_oracle://' + config.PATSTAT_USER_COMMON + ':' + config.PATSTAT_PASSWORD_COMMON + '@ora4.zew-private.de:1521/test' # [DB_FLAVOR]+[DB_PYTHON_LIBRARY]://[USERNAME]:[PASSWORD]@[DB_HOST]:[PORT]/[DB_NAME] - adjust as needed
engine = create_engine(patstat_connection_string) # engine leads to PATSTAT

# Check whether connection has been successfully established.

pd.read_sql("SELECT * FROM all_synonyms WHERE table_owner = 'PAT2'", engine).head(3)

# ## All patents of German firms 

# In the following aplication we analyze PATSTAT patent applications from companies found in the Mannheim Enterprise Panel (MUP).

# Read match between EPO and MUP entities
df_epo2vvc = pd.read_csv(here("./01_Data/01_Patents/epo2vvc2019best.txt"), sep='\t')

# Capitalize column names
df_epo2vvc.columns = [x.upper() for x in df_epo2vvc.columns]

df_epo2vvc.head(3)

# How many distinct patent applications are there?
len(df_epo2vvc.APPLN_ID.drop_duplicates())

# How many distinct patent applicants (companies) are there?
len(df_epo2vvc.CREFO.drop_duplicates())

# %%time
# Write df into database
df_epo2vvc[['APPLN_ID']].drop_duplicates().reset_index(drop=True).to_sql(
    'EPO2VVC'.lower(),
    engine,
    if_exists='replace',
    index=False,
    chunksize=500,
    dtype={
        "APPLN_ID": types.VARCHAR(20),
    }
)

# Create index to make everything faster 
with engine.connect() as connection:
    connection.execute('create index epo2vvc_ix on EPO2VVC(appln_id)')

# %%time
# Create df with patent applications of MUP firms and the respective CPC classes
df_cpcs = pd.read_sql('select * from epo2vvc t1 left join A20224_APPLN_CPC t2 on t1.appln_id=t2.appln_id', engine)

# Save results for later usage. Note this dataset has long format, i.e. one row presents one of (the possibly many) CPC classes attached to a patent application
df_cpcs.iloc[:,[0,2]].to_csv(here("./01_Data/01_Patents/epo2vvc_cpc_classes.txt"), sep='\t', encoding='utf-8', index=False)

# %%time
# Create df with patent applications of MUP firms and the respective abstracts
df_abs = pd.read_sql('SELECT * FROM epo2vvc t1 LEFT JOIN A20203_APPLN_ABSTR t2 ON t1.appln_id=t2.appln_id', engine)

# Capitalize column names
df_abs.columns = [x.upper() for x in df_abs.columns]

df_abs.shape

df_abs.head(3)

# Abstract names are spread across at most 4 columns in the Oracle database. The following lines of code join these columns into one column representing the patent abstract of the respective patent application.

# Reduce df to the relevant variables only
df_abs = df_abs[['APPLN_ID','TEIL1','TEIL2','TEIL3','TEIL4','APPLN_ABSTRACT_LG','LEN_ABSTRACT']]
df_abs = df_abs.iloc[:,[0, 2, 3, 4, 5, 6, 7]]

df_abs.head(3)

# Merge 4 columns into one column considering that not all abstracts spread over all 4 columns
from tqdm import tqdm
temp = []
for i in tqdm(range(0, len(df_abs))):
    df_temp = df_abs.iloc[i]
    if pd.notna(df_temp.TEIL4):
        temp.append((df_temp.APPLN_ID, (df_temp.TEIL1 + ' ' + df_temp.TEIL2 + ' ' + df_temp.TEIL3 + ' ' + df_temp.TEIL4)))
    elif pd.notna(df_temp.TEIL3):
        temp.append((df_temp.APPLN_ID, (df_temp.TEIL1 + ' ' + df_temp.TEIL2 + ' ' + df_temp.TEIL3)))
    elif pd.notna(df_temp.TEIL2):
        temp.append((df_temp.APPLN_ID, (df_temp.TEIL1 + ' ' + df_temp.TEIL2)))
    else:
        temp.append((df_temp.APPLN_ID, df_temp.TEIL1))

# Create temporary df
df_temp = pd.DataFrame(temp, columns=['APPLN_ID', 'ABSTRACT'])

df_abs = df_abs.rename(columns = {'APPLN_ABSTRACT_LG': 'ABSTRACT_LANG', 'LEN_ABSTRACT': 'ABSTRACT_LEN'})

# Merge temporary df back to main df - now with with patent abstracts only in one column
df_abs = df_abs.merge(df_temp, on = 'APPLN_ID')
df_abs = df_abs[['APPLN_ID', 'ABSTRACT', 'ABSTRACT_LANG', 'ABSTRACT_LEN']]

# Some abstract texts are not in English. Translate patent abstracts in any other language into English.

df_abs.ABSTRACT_LANG.value_counts(dropna=False)

# Number of non-English abstracts
df_temp.shape

df_temp = df_abs.loc[(df_abs.ABSTRACT_LANG!='en') & (df_abs.ABSTRACT_LANG.notnull()),:].set_index('APPLN_ID', drop=True)

df_temp.head(3)


# ## All Y02 patents 

# Alternatively, all Y02 patents that have been filed at EPO can be selected as corpus basis.

# +
# Deprecated
# -

# # Clean data 

# The problem with the non-English abstracts is that their encoding (umlaute, accents, etc.) is erroneous. So, first we need to get the encoding right.

# Function to correct encoding
def clean_encoding(x):
    try:
        return(x.encode("windows-1252").decode("utf-8"))
    except:
        return(x)


# Correct encoding
df_temp['ABSTRACT'] = df_temp.ABSTRACT.apply(lambda x: clean_encoding(x))

df_temp.head(3)

# Load function translation_handler() which relies on module deep_translator in order to translate abstracts
from util import translation_handler


def patent_translation(df_temp):
    temp = list()
    for i in range(len(df_temp)):
        temp.append((df_temp.index[i], translation_handler(df_temp.iloc[i]['ABSTRACT'])))
    return(temp)


n = len(df_temp)
chunk_size = 100
for start in tqdm(range(0, n, chunk_size)):
    df_subset = df_temp.iloc[start:(start+chunk_size)]
    patent_translations = patent_translation(df_subset)
    with open(here('02_Code/.pycache/patent_translations.pkl'), 'ab+') as f:
        f.write(pkl.dumps(patent_translations))

from util import read_cache

temp = read_cache(here('02_Code/.pycache/patent_translations.pkl'))

len(temp)

# Create temporary df
df_temp = pd.DataFrame(temp, columns=['APPLN_ID', 'ABSTRACT'])

# Merge temporary df back to main df - now with translated patent abstracts
df_temp = df_abs.merge(df_temp, on='APPLN_ID', how='left')

df_temp['ABSTRACT'] = np.where(df_temp['ABSTRACT_y'].isnull(), df_temp['ABSTRACT_x'], df_temp['ABSTRACT_y'])

df_abs = df_temp[['APPLN_ID', 'ABSTRACT', 'ABSTRACT_LANG', 'ABSTRACT_LEN']]

df_abs.shape

df_abs

# + [markdown] tags=[]
# ## Aggregate Data 
# -

# According to EPO, the Cooperative Patent Classification (CPC) is an extension of the IPC and is jointly managed by the EPO and the US Patent and Trademark Office.
# We now aggregate the CPC information both at patent level and later at firm level to get an idea to which technological fields the patents relate to and to proxy the technological profiles of firms.

# + [markdown] tags=[]
# ### CPC Level 
# -

df_cpcs = pd.read_csv(here("./01_Data/01_Patents/epo2vvc_cpc_classes.txt"), sep='\t', encoding='utf-8')

df_cpcs.shape

df_cpcs.head(3)

# Uppercase column names and convert APPLN_ID as integer variable
df_cpc = df_cpcs.copy()
df_cpc.columns = [x.upper() for x in df_cpc.columns]

n1 = len(df_cpc.drop_duplicates('APPLN_ID'))

# Drop patents w/o CPC class
df_cpc = df_cpc.loc[df_cpc.CPC_CLASS_SYMBOL.notnull(),:]

n2 = len(df_cpc.drop_duplicates('APPLN_ID'))
n1 - n2, round((n1 - n2)/n1, 5)

# For 177 patents, i.e. less than 0.1% no CPC information exists. This is negligible.

# CPC is divided into nine sections, A-H and Y, which in turn are sub-divided into classes, sub-classes, groups and sub-groups. There are approximately 250 000 classification entries. For now, I am only interested in the sections for A-H. In case of section Y, I am particularly interested in the Y02 class. According to Angelucci et al. ([2018](https://www.sciencedirect.com/science/article/pii/S0172219016300618)), EPO has developed a dedicated classification scheme for CPC. This classification scheme is the starting point for developing a NLP model capable of identifying cleantech firms.

# Extract CPC section from CPC class information 
import re
df_cpc['CPC'] = df_cpc.CPC_CLASS_SYMBOL.apply(lambda x: re.search(r'A|B|C|D|E|F|G|H|(Y02\w{1})|(Y\d\d)', x).group(0))

df_cpc.CPC.value_counts(dropna=False)

# Remove redundant whitespaces in CPC_CLASS_SYMBOL
df_cpc['CPC_CLASS_SYMBOL'] = df_cpc.CPC_CLASS_SYMBOL.apply(lambda x: re.sub(' +', ' ', x))

# Get CPC group
df_cpc['CPC_GROUP'] = df_cpc.CPC_CLASS_SYMBOL.apply(lambda x: re.sub(r'/\d{1,}', '', x))

# Create seperate Y02 class column
df_cpc['Y02'] = df_cpc.CPC_CLASS_SYMBOL.apply(lambda x: re.search(r'Y02\w{1}', x).group(0) if re.search(r'Y02\w{1}', x) else np.nan)

df_cpc.Y02.value_counts(dropna=False)

# Alternatively, it seems to be more practical to create clean-tech market indicators following the mapping in the file below.

df_map = pd.read_excel(here('./01_Data/02_Firms/df_cleantech_firms_label.xlsx'), sheet_name='Tag2CPC')

# Create a mapping dictionary from CLEANTECH_MARKET to relevant CPCs
df_map = df_map.loc[df_map.ABBREVIATION.notnull() & df_map.CPC.notnull(),]
tech2market = dict(zip(df_map['CPC'], df_map['ABBREVIATION']))

# First map all detailed CPC class symbols, second map the remaining according the cpc group, third fill the CPC class to the non-cleantech lines
df_cpc['CLEANTECH_MARKET'] = df_cpc.CPC_CLASS_SYMBOL.map(tech2market)
df_cpc.loc[df_cpc['CLEANTECH_MARKET'].isnull(), 'CLEANTECH_MARKET'] = df_cpc.loc[df_cpc['CLEANTECH_MARKET'].isnull(), 'CPC_GROUP'].map(tech2market)
df_cpc.loc[df_cpc['CLEANTECH_MARKET'].isnull(), 'CLEANTECH_MARKET'] = df_cpc.loc[df_cpc['CLEANTECH_MARKET'].isnull(), 'CPC'].map(tech2market)
df_cpc.loc[df_cpc['CLEANTECH_MARKET'].isnull(), 'CLEANTECH_MARKET'] = df_cpc.loc[df_cpc['CLEANTECH_MARKET'].isnull(), 'CPC']

# Drop duplicates in APPLN_ID, CPC, Y02 and CLEANTECH_MARKET
df_cpc = df_cpc[['APPLN_ID', 'CPC', 'CLEANTECH_MARKET']].drop_duplicates().reset_index(drop=True)

df_cpc.shape

# Create seperate cleantech class column
df_cpc['Y02'] = df_cpc.CLEANTECH_MARKET.apply(lambda x: 1 if x in ['Water', 'E-Efficiency', 'Generation', 'Biofuels', 'Materials', 'Adaption', 'CCS', 'Grid', 'ICT', 'Battery', 'E-Mobility'] else np.nan)

# For aggregating from CPC level to patent level we want to know how much percent of the patent relates to which CPC class.

from collections import Counter
Counter(['Generation', 'Generation', 'Biofuels', 'A', 'B', 'C'])


# This function calculates the relative importance of the CPC class for the patent
def counter_to_relative(x):
    counter = Counter(x)
    total_count = sum(counter.values())
    relative = {}
    for key in counter:
        relative[key] = counter[key] / total_count
    return relative


counter_to_relative(['Generation', 'Generation', 'Biofuels', 'A', 'B', 'C'])


# This function extracts the relative importance of the Y02 classes only
def agg_cpc(x):
    y02s = ['Water', 'E-Efficiency', 'Generation', 'Biofuels', 'Materials', 'Adaption', 'CCS', 'Grid', 'ICT', 'Battery', 'E-Mobility']
    if any([cpc for cpc in x if cpc in y02s]):
        relative = counter_to_relative(x)
        relative_y02 = {i: relative[i] for i in y02s if relative.get(i) is not None}
        return relative_y02
    else:
        return {}


agg_cpc(['Generation', 'Generation', 'Biofuels', 'A', 'B', 'C'])

# ### Patent Level 

# Aggregate patent data from CPC level to patent level.

# Apply agg_cpc()
df_pat = df_cpc.groupby('APPLN_ID').agg({
    'CPC': lambda x: list(set(x)),
    'CLEANTECH_MARKET': lambda x: list(set(x))}).reset_index()
df_pat['Y02_dict'] = df_pat.CLEANTECH_MARKET.apply(lambda x: agg_cpc(x))

df_pat.shape

# Calculate the overall relative importance of Y02 (sum over values in Y02 dict) 
df_pat['Y02_imp'] = df_pat.Y02_dict.apply(lambda x: sum(x.values()))

# Look at distribution of Y02 importance
df_pat.Y02_imp.value_counts()

# Create Y02 identifier
df_pat['Y02'] = df_pat.Y02_imp.apply(lambda x: 1 if x > 0 else 0)

df_pat.Y02.value_counts(normalize=True)

# 7% of patents are cleantech-related patents.

# Merge abstract texts

# Read patent data
df_abs = pd.read_pickle(here(r'.\01_Data\01_Patents\epo2vvc_patents.pkl'))
df_abs.shape

df_abs=df_abs[['APPLN_ID', 'ABSTRACT_LANG' ,'ABSTRACT_LEN', 'ABSTRACT', 'LEMMAS']]

df_abs.dtypes

# Adjust dtype of ID variable in df_pat
df_pat['APPLN_ID'] = df_pat.APPLN_ID.astype(str)

df_pat.shape

df_pat.dtypes

# Merge abstracts to df_pat
df_pat = df_pat.merge(df_abs, on='APPLN_ID', how='left')

df_pat.shape

df_pat.head(3)

# Conduct lemmatization and stopword deletion of the patent abstracts. This step can be useful for the later model development.

# Load lemmatizer form util.py and make pandas apply() function showing a progress bar
from util import string_to_lemma
tqdm.pandas()

n1 = len(df_pat.drop_duplicates('APPLN_ID'))
n2 = df_pat.ABSTRACT.isnull().sum()
n2, round((n2)/n1, 5)

# For 1206 patents, i.e. less than 0.25% no abstract text exists. This is negligible.

df_pat = df_pat.loc[df_pat.ABSTRACT.notnull()]

df_pat.shape

df_pat['LEMMAS'] = df_pat.ABSTRACT.progress_apply(lambda x: string_to_lemma(x))

df_pat.loc[df_pat.CPC.apply(lambda x: 'Y02E' in x), ['CPC', 'ABSTRACT']].sample(3).style.set_properties(subset=['ABSTRACT'], **{'width': '900px'})

len(df_pat)

# Corpus consists of close to 560,000 patents.

# Count number of lemmas
vocs = set()
for index, row in tqdm(df_pat.iterrows()):
    voc = set(row.LEMMAS)
    vocs.update(voc)
#    if index > 100:
#        break
len(vocs)

df_pat.to_pickle(here(r".\01_Data\01_Patents\epo2vvc_patents.pkl"))

# ### Firm Level 

# Aggregate data from patent level to firm level.

# Read match between EPO and MUP entities
df_epo2vvc = pd.read_csv(here("./01_Data/01_Patents/epo2vvc2019best.txt"), sep='\t')

# Read file with firm info (most importantly firm age)
df_dates_w58 = pd.read_stata(config.,PATH_TO_MUP + "/Daten/Aufbereitet/gruenddat_aufb_w58.dta")

# Read prepared patent data
df_pat = pd.read_pickle(here("./01_Data/01_Patents/epo2vvc_patents.pkl"))

colnames = ['appln_id', 'crefo', 'appln_fili', 'earliest_f', 'granted']
df_temp = df_epo2vvc[colnames]
df_temp.columns = [col.upper() for col in colnames]
df_temp['APPLN_ID'] = df_temp.APPLN_ID.astype(str)

# Merge crefo (firm identifier) earliest_f (year of filing the patent) granted (information whether patent has been granted) to df_pat
df_pat = df_pat.merge(df_temp, how='left', on='APPLN_ID')

# Drop columns which are not of interest
df_pat = df_pat[['APPLN_ID', 'CREFO', 'Y02', 'APPLN_FILI', 'EARLIEST_F', 'GRANTED']]

df_pat.head(3)

# Extract distinct company IDs and look at number of patenting firms
crefos = list(df_pat.CREFO.drop_duplicates().values)
len(crefos)

df_pat.dtypes

# Do some cleaning in df_dates_w58
df_dates_w58['crefo'] = df_dates_w58.crefo.astype(pd.Int64Dtype()).astype(np.int64)
df_temp = df_dates_w58.loc[df_dates_w58.crefo.isin(crefos)]
df_temp.columns = [col.upper() for col in df_temp.columns]

df_temp.GRUEND_JAHR.notnull().sum()/len(df_temp)

# For 96% of the patenting firms, the founding year exists. The missing 4% can be neglected for a quick look at the distribution of the age of firms at time of patent filing.

# Now merge firm information
df_pat = df_pat.merge(df_temp[['CREFO', 'GRUEND_JAHR']], on='CREFO', how='left')

# Calculate age at filing patent
df_pat['FILING_JAHR'] = pd.DatetimeIndex(df_pat.APPLN_FILI).year
df_pat['AGE_AT_FILING'] = df_pat.FILING_JAHR-df_pat.GRUEND_JAHR

df_pat.loc[(df_pat.AGE_AT_FILING>0) & (df_pat.AGE_AT_FILING<300),:].AGE_AT_FILING.plot(kind='hist', bins=100)

# Distribution looks pretty skewed but not in the direction as expected.

# Continue here with additional firm characteristics and subsequent aggregation to firm level.

# Now conduct aggregation from patent level to firm level
df_firm = df_epo2vvc.groupby('CREFO').agg(
    {'PERSON_ID': lambda x: list(set(x)),
     'APPLN_ID': lambda x: list(set(x)),
     'APPLN_NR': lambda x: list(set(x)),
     'APPLN_FILI': lambda x: list(set(x)),
     'INPADOC_FA': lambda x: list(set(x)),
     'EARLIEST_F': lambda x: list(x),
     'INPADOC_FA': lambda x: list(set(x)),
     'GRANTED': lambda x: x.mean(),
     'CPC': lambda x: [i for j in x for i in j],
     #'Y02_imp': lambda x: x.mean(),
     'ABSTRACT': lambda x: ' '.join(x)
    }
)

# Add additional information
df_firm['N_PATENTS'] = df_firm.EARLIEST_F.apply(len)                       # number of patent apllications by firm
df_firm['Y02_dict'] = df_firm.CPC.apply(lambda x: agg_cpc(x))              # importance of each of the Y02 classes
df_firm['Y02_imp'] = df_firm.Y02_dict.apply(lambda x: sum(x.values()))     # overall importance of Y02
df_firm['Y02'] = df_firm.Y02_imp.apply(lambda x: 1 if x > 0 else 0)        # create Y02 identifier

df_firm.Y02.value_counts(normalize=True)

# 16% of firms have applied for a clean-tech related patent.

# Think further how aggregation is done best here:
# - abstracts only of Y02 patents?
# - filing dates?
# - ...

df_firm.head(3)
