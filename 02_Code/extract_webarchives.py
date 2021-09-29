# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Extract website content of the test sample of cleantech and NASDAQ firms from web archives.

import pandas as pd
import re
from util import View
from pyprojroot import here

# +
# Read test data of cleantech and nasdaq firms
df_cleantech = pd.read_csv(here('01_Data/02_Firms/df_cleantech_firms.txt'), sep='\t', encoding='utf-8')
df_cleantech = df_cleantech[['NAME', 'WEBSITE']]
df_cleantech['LABEL'] = 'cleantech'

df_nasdaq = pd.read_csv(here('01_Data/02_Firms/df_nasdaq_firms.txt'), sep='\t', encoding='utf-8')
df_nasdaq = df_nasdaq[['NAME', 'WEBSITE_LINK']].rename(columns={'WEBSITE_LINK': 'WEBSITE'})
df_nasdaq['LABEL'] = 'nasdaq'

# Combine both firm samples in one df
df_url = pd.concat([df_cleantech, df_nasdaq]).reset_index(drop=True)


# -

# Function to clean urls
def wildcarding(url):
    domain_end = r'(?:com|de|net|io|co|energy|ai|life|ac|ly|xyz|us)'
    try:
        url_wildcarded = re.search(r'\..{1,}\.' + domain_end, url).group(0)[1:] + '/*'
    except AttributeError:
        try:
            url_wildcarded = re.search(r'.{1,}\.' + domain_end, url).group(0) + '/*'
        except:
            try:
                url_wildcarded = re.search(r'.{1,}-(?:com|de)', url).group(0)[1:] + '/*'
            except:
                url_wildcarded = url + '/*'
        url_wildcarded = re.sub(r'(http://)|(https://)', '', url_wildcarded)
    return url_wildcarded


# +
# All urls are "clean" if this cell equals 0
ind = []
for i in range(df_url.shape[0]):
    if pd.notna(df_url.WEBSITE[i]):
        try:
            wildcarding(df_url.WEBSITE[i])
        except:
            ind.append(i)

len(ind)
# -

# Clean urls (wildcarding)
df_url['WEBSITE_CLEAN'] = df_url.loc[df_url.WEBSITE.notnull(), 'WEBSITE'].apply(lambda x: wildcarding(x))
df_url = df_url[['NAME', 'LABEL', 'WEBSITE', 'WEBSITE_CLEAN']]

df_url.head(3)

# Extract archived versions of the respective corporate websites.

import cdx_toolkit
from tqdm import tqdm

client = cdx_toolkit.CDXFetcher(source='ia')       # define client for fetching data from source (ia: Internet Archive, cc: Common Crawl)
source = 'ia'                                      # web archive to extract data from ('ia': Internat Archive, 'cc': Common Crawl)
limit = 50                                         # define maximum number of captures that is suppossed to be retrieved for each year-url from the respective archive
firms = list(df_url.NAME.drop_duplicates().values) # create list of unique company IDs (crefos) for which panel dataset of corporate website content is created
len(firms)

# Create object for writing archive captures into .warc files
writer = cdx_toolkit.warc.CDXToolkitWARCWriter(
        prefix='cleantech_',      # first part of .warc file where warc records will be stored
        subprefix=source,     # second part of .warc file where warc records will be stored
        info=warcinfo,           
        size=1000000000,         # once the .warc file exceeds 1 GB of size a new .warc file will be created for succeeding records
        gzip=True)

# A 'warcinfo' record describes the records that follow it, up through end of file, end of input, or until next 'warcinfo' record.
# Typically, this appears once and at the beginning of a WARC file. 
# For a web archive, it often contains information about the web crawl which generated the following records.
warcinfo = {
    'software': 'pypi_cdx_toolkit iter-and-warc',
    'isPartOf': 'CLEANTECH_NASDAQ',
    'description': 'warc extraction',
    'format': 'WARC file version 1.0',
}

# Set directory for cdx fetcher to write
import os 
os.chdir(here('01_Data/02_Firms/01_Webdata/'))

# + tags=[] jupyter={"outputs_hidden": true}
# %%time
for year in range(2010, 2022):
    # Create control files if not already exist
    open(here('01_Data/02_Firms/01_Webdata/' + 'captured-' + str(year) + '.txt'), 'a+').close()
    open(here('01_Data/02_Firms/01_Webdata/' + 'not_captured-' + str(year) + '.txt'), 'a+').close()
    open(here('01_Data/02_Firms/01_Webdata/' + 'http_error-' + str(year) + '.txt'), 'a+').close()
    
    # Read list of already accessed firms
    # Captured by web archive:
    with open(here('01_Data/02_Firms/01_Webdata/' + 'captured-' + str(year) + '.txt'), 'r+') as f:
        captured = f.read().splitlines()
    # Not captured by web archive:
    with open(here('01_Data/02_Firms/01_Webdata/' + 'not_captured-' + str(year) + '.txt'), 'r+') as f:
        not_captured = f.read().splitlines()
    # Extraction attempt reulted in http error:
    with open(here('01_Data/02_Firms/01_Webdata/' + 'http_error-' + str(year) + '.txt'), 'r+') as f:
        http_error = f.read().splitlines()
    captured.extend(not_captured)
    
#    # Create object for writing archive captures into .warc files
#    writer = cdx_toolkit.warc.CDXToolkitWARCWriter(
#        prefix='cleantech',      # first part of .warc file where warc records will be stored
#        subprefix=str(year),     # second part of .warc file where warc records will be stored
#        info=warcinfo,           
#        size=1000000000,         # once the .warc file exceeds 1 GB of size a new .warc file will be created for succeeding records
#        gzip=True)
    
    for i, firm in enumerate(firms):
        # Pass if crefos has already been accessed
        if firm in captured:
            print("Done:", firm)
            continue
        
        row = df_url.loc[(df_url.NAME==firm),:].squeeze(axis=0)
        
        if pd.isna(row.WEBSITE_CLEAN):
            pass                 # pass if firm has not existed in the respective year (which refers to the url entry is NA in the respective year)
        else:
            print(str(i), '- Firm: ', str(row.NAME))
            try:
                # Create iterator object including all captures in of the url in the given year
                capture = client.iter(row.WEBSITE_CLEAN, from_ts=str(year), to=str(year), limit=limit, verbose='v', collapse='urlkey', filter=['status:200', 'mime:text/html'])
                
                # If corporate website has not been captured write crefo to "not_captured-YEAR.txt" file
                if len(capture.captures) == 0:
                    with open(here('01_Data/02_Firms/01_Webdata/' + 'captured-' + str(year) + '.txt'), 'a') as name_out:
                            name_out.write("%s\n" % firm)
                
                # Else iterate over all captures and save header and body information in "ID.warc.gz" file
                else:
                    with open(here('01_Data/02_Firms/01_Webdata/' + 'not_captured-' + str(year) + '.txt'), 'a') as name_out:
                            name_out.write("%s\n" % firm)
                    for obj in tqdm(capture):
                        url = obj['url']
                        status = obj['status']
                        timestamp = obj['timestamp']

                        try:
                            record = obj.fetch_warc_record()
                            # Save crefo into header information of the WARC record so it is not lost in the WARC file
                            record.rec_headers['name'] = str(row.NAME)
                            writer.write_record(record)
                        
                        # Single captures can run into errors:
                        # Except RuntimeError
                        except RuntimeError:
                            print('Skipping capture for RuntimeError 404: %s %s', url, timestamp)
                            continue                
                    
                        # Except encoding error that typically arises if no content found on webpage
                        except UnicodeEncodeError:
                            print('Skipping capture for UnicodeEncodeError: %s %s', url, timestamp)
                            continue
            
            # URLs can also run into errors
            # Except HTTPError if URL has been excluded from the Wayback Machine
            except requests.HTTPError:
                print('Skipping url for HTTPError 403: %s', NAME)
                with open(here('01_Data/02_Firms/01_Webdata/' + 'http_error-' + str(year) + '.txt'), 'a') as name_out:
                            name_out.write("%s\n" % firm)
                continue
# Close                 
writer.fd.close()
# -

# Execute this cell, if you you want to close the CDXToolkitWARCWriter 
writer.fd.close()
