# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, rvest, janitor, memoise, here, rjson, pbapply)

# Under this link you get the base table from https://www.cnbc.com/nasdaq-100/ in json format
link <- "quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?symbols=AMD%7CADBE%7CALGN%7CAMZN%7CAMGN%7CAEP%7CADI%7CANSS%7CAAPL%7CAMAT%7CASML%7CTEAM%7CADSK%7CATVI%7CADP%7CAVGO%7CBIDU%7CBIIB%7CBMRN%7CBKNG%7CCDNS%7CCDW%7CCERN%7CCHKP%7CCHTR%7CCPRT%7CCRWD%7CCTAS%7CCSCO%7CCMCSA%7CCOST%7CCSX%7CCTSH%7CDOCU%7CDXCM%7CDLTR%7CEA%7CEBAY%7CEXC%7CFAST%7CFB%7CFISV%7CFOX%7CFOXA%7CGILD%7CGOOG%7CGOOGL%7CHON%7CILMN%7CINCY%7CINTC%7CINTU%7CISRG%7CMRVL%7CIDXX%7CJD%7CKDP%7CKLAC%7CKHC%7CLRCX%7CLULU%7CMELI%7CMAR%7CMTCH%7CMCHP%7CMDLZ%7CMRNA%7CMNST%7CMSFT%7CMU%7CNFLX%7CNTES%7CNVDA%7CNXPI%7COKTA%7CORLY%7CPAYX%7CPCAR%7CPDD%7CPTON%7CPYPL%7CPEP%7CQCOM%7CREGN%7CROST%7CSIRI%7CSGEN%7CSPLK%7CSWKS%7CSBUX%7CSNPS%7CTCOM%7CTSLA%7CTXN%7CTMUS%7CVRSN%7CVRSK%7CVRTX%7CWBA%7CWDAY%7CXEL%7CXLNX%7CZM&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1&output=json&events=1"


# Scrape base table of companies in 2021 Global Cleantech 100 List
parse_tab <- function(data){
  
  data <- data$FormattedQuoteResult$FormattedQuote
  n_firms <- length(data)
  tab <- lapply(1:n_firms, function(firm) data[[firm]][1:22]) %>% 
    bind_rows() %>% 
    clean_names("all_caps") %>% 
    mutate(HREF = paste0("https://www.cnbc.com/quotes/", SYMBOL, "?tab=profile")) %>% 
    select(SYMBOL, HREF, everything())
  return(tab)

}


# Auxiliary functions:
# Simple string cleaner (1)
clean_pattern <- function(string){
  return(gsub(pattern = '"|:', '', string))
}

# Simple string cleaner (2)
clean_entry <- function(string){
  return(gsub(pattern = '\\[\"|\"|\",|\"\\]', '', string))
}

# Extract firm-specific database entries from string
extract_entries <- function(string, pattern_vector){
  entry_list <- list()
  n_patterns <- length(pattern_vector)
  for (i in 1:(n_patterns-1)){
    
    pattern <- pattern_vector[i]
    start <- str_locate_all(string, pattern = regex(pattern))[[1]][2]
    end <- str_locate_all(string, pattern = regex(pattern_vector[i+1]))[[1]][1]
    entry <- substr(string, start+1, end-2)
    
    # Seperation of address information 
    # DEPRECATED!
    if (pattern=='\"address\":'){
      address_fields <- c('street', 'city', 'zip_code', 'country')
      
      temp <- lapply(str_split(entry, "\","), clean_entry)[[1]]
      for (i in 1:length(temp)){
        entry_list[address_fields[i]] <- temp[i]
      }
    }
    
    else{
      entry_list[[clean_pattern(pattern)]] <- clean_entry(entry)  
    }
    
  }
  
  return(entry_list)
  
}


# Scrape detailed firm info
scrape_firminfo <- function(url, pattern_vector){
  text <- read_html(url) %>%
    html_text2()
  
  start <- str_locate_all(string = text, pattern = regex('\"companyData\":'))[[1]][1]
  end <- str_locate_all(string = text, pattern = regex('\\[\\{\"Designation\":'))[[1]][1]
  company_text <- substr(text, start, end-1)
  entry_row <- extract_entries(company_text, pattern_vector) %>% as_tibble()

  
  return(entry_row)
}


# Cache function scrape_firminfo
# Define cache_dir inside the function
cache_dir <- here("02_Code/.rcache")
## Create this directory if it doesn't yet exist
if (!dir.exists(cache_dir)) dir.create(cache_dir)

cached_scrape_firminfo <- memoise(scrape_firminfo, cache = cache_filesystem(cache_dir))



# Scraping ----------------------------------------------------------------

## NASDAQ 100 =============================================================


# First, scrape the base table 
data <- fromJSON(paste(readLines(here('01_Data/02_Firms/nasdaq.json')), collapse=""))

base_table <- parse_tab(data)

# Second, given the HREF in the base table scrape the detailed firm information

pattern_vector <- c(
  '\"industryClassification\":',
  '\"businessSummary\":',
  '\"address\":',
  '\"websiteLink\":',
  '\"officers\":'
)



df_nasdaq <- pblapply(base_table$HREF, function(url){
  tryCatch(cached_scrape_firminfo(url, pattern_vector) %>% bind_cols("HREF" = url), error=function(e) NULL)
}) %>% bind_rows()



# Clean data
df_nasdaq <- df_nasdaq %>% 
  mutate(websiteLink = str_replace_all(websiteLink, pattern = "\\\\u002F", replacement = "/")) %>% 
  clean_names("all_caps") %>% 
  left_join(base_table, by = "HREF") %>% 
  type_convert() %>% 
  select(NAME, WEBSITE_LINK, BUSINESS_SUMMARY, everything())

# Save to file
df_nasdaq %>% 
  write_delim(here("01_Data/02_Firms/df_nasdaq_firms.txt"), delim = '\t')



## NASDAQ all =============================================================

base_table_nasdaq <- read_csv(here("01_Data/02_Firms/nasdaq.csv"))
base_table_nasdaq <- base_table_nasdaq %>% 
  mutate(NASDAQ=1) %>% 
  select(Symbol, NASDAQ)
base_table <- read_csv(here("01_Data/02_Firms/nasdaq_nyse_amex.csv"))
base_table <- base_table %>% left_join(base_table_nasdaq)

pattern_vector <- c(
  '\"businessSummary\":',
  '\"industryClassification\":',
  '\"websiteLink\":',
  '\"address\":',
  '\"officers\":'
)

df_nasdaq <- pblapply(base_table$Symbol, function(url){
  tryCatch(cached_scrape_firminfo(url, pattern_vector) %>% bind_cols("Symbol" = url), error=function(e) NULL)
}) %>% bind_rows()


# Clean data
df_nasdaq <- df_nasdaq %>% 
  mutate(websiteLink = str_replace_all(websiteLink, pattern = "\\\\u002F", replacement = "/")) %>% 
  clean_names("all_caps") %>% 
  type_convert() %>% 
  select(SYMBOL, WEBSITE_LINK, BUSINESS_SUMMARY, everything())

# Add business summaries to base table
df_nasdaq <- base_table %>%
  clean_names("all_caps") %>% 
  left_join(df_nasdaq) %>% 
  select(SYMBOL, NAME, BUSINESS_SUMMARY, WEBSITE_LINK, NASDAQ, everything()) 

# Save to file
df_nasdaq %>% 
  write_delim(here("01_Data/02_Firms/df_all_stocks_firms.txt"), delim = '\t')
