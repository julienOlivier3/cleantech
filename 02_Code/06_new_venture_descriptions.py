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

# # Create list of German company foundations 

# We define a new venture as not being older than 10 years.

df = pd.read_stata(r"K:\MUP\Paneldaten\Daten\Aufbereitet\gruenddat_w60_teil_8.dta")

df.head(3)

df = pd.DataFrame()
for i in tqdm(range(2, 9)):
    df_temp = pd.read_stata(
        r"K:\MUP\Paneldaten\Daten\Aufbereitet\gruenddat_w59_teil_" + str(i) +
        r".dta",
        preserve_dtypes=False)
    # Filter rows with missing crefo (if exist)
    df_temp = df_temp.loc[df_temp.crefo.notnull(),
                          ['crefo', 'gruenddat', 'gruend_jahr']]
    # Filter to firms at most 10 years old
    df_temp = df_temp.loc[df_temp.gruend_jahr >= 2012, :]
    # Convert data types of variables
    df_temp = df_temp.convert_dtypes()
    df_temp['crefo'] = df_temp.crefo.astype(np.int64)

    # Append df_temp to df
    df = df.append(df_temp)
