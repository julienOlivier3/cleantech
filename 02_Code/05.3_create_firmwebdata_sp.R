if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, rvest, janitor, memoise, here, rjson, pbapply)

landing <- "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# Scrape base table of companies in 2021 Global Cleantech 100 List
scrape_tab <- function(url){
  
  html <- read_html(url)
  
  tab <- html %>% 
    html_table() %>% 
    .[[1]]
  
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
scrape_firminfo <- function(symbol, pattern_vector){
  url <- paste0("https://www.cnbc.com/quotes/", symbol, "?tab=profile")
  text <- read_html(url) %>%
    html_text2()
  
  start <- str_locate_all(string = text, pattern = regex('\"companyData\":'))[[1]][1]
  end <- str_locate_all(string = text, pattern = regex('\\[\\{\"Designation\":'))[[1]][1]
  company_text <- substr(text, start, end-1)
  entry_row <- extract_entries(company_text, pattern_vector) %>% as_tibble()
  
  
  return(entry_row)
}


# Scrape detailed firm info from html
scrape_firminfo_html <- function(symbol){
  html <- read_html(paste0("https://apps.cnbc.com/view.asp?symbol=", symbol, "&uid=stocks/summary"))
  
  html_reduced <- html %>% 
    html_element('body') %>% 
    html_element(xpath = '//div[@class="module"]') %>% 
    html_element('div') %>% 
    html_elements('table')
  
  entry_list <- list()
  
  entry_list["BUSINESS_SUMMARY"] <- html_reduced[[1]] %>% 
    html_elements('td') %>% 
    html_element(xpath = '//td[@class="desc"]') %>% 
    .[[1]] %>% 
    html_elements('div') %>% 
    html_text() %>% 
    .[2]
  
  entry_list["WEBSITE"] <- html_reduced[[3]] %>%
    html_element('td') %>% 
    html_element('tr') %>% 
    html_elements('a') %>% 
    html_attr('href')
  
  return(entry_list %>% as_tibble())
  
}




# Cache function scrape_firminfo
# Define cache_dir inside the function
cache_dir <- here("02_Code/.rcache")
## Create this directory if it doesn't yet exist
if (!dir.exists(cache_dir)) dir.create(cache_dir)

cached_scrape_firminfo <- memoise(scrape_firminfo, cache = cache_filesystem(cache_dir))
cached_scrape_firminfo_html <- memoise(scrape_firminfo_html, cache = cache_filesystem(cache_dir))


base_table <- scrape_tab(landing)


pattern_vector <- c(
  '\"businessSummary\":',
  '\"industryClassification\":',
  '\"websiteLink\":',
  '\"address\":',
  '\"officers\":'
)

df_sp <- pblapply(base_table$Symbol, function(symbol){
  tryCatch(cached_scrape_firminfo(symbol, pattern_vector) %>% bind_cols("Symbol" = url), error=function(e) NULL)
}) %>% bind_rows()


# Clean data
df_sp <- df_sp %>% 
  mutate(websiteLink = str_replace_all(websiteLink, pattern = "\\\\u002F", replacement = "/")) %>% 
  left_join(base_table, by = "Symbol") %>% 
  clean_names("all_caps") %>% 
  type_convert() %>% 
  select(SECURITY, WEBSITE_LINK, BUSINESS_SUMMARY, everything())

# Save to file
df_sp %>% 
  write_delim(here("01_Data/02_Firms/df_sp_firms.txt"), delim = '\t')

# Second version with better coverage of business descriptions
df_sp <- pblapply(base_table$Symbol, function(symbol){
  tryCatch(cached_scrape_firminfo_html(symbol) %>% bind_cols("Symbol" = symbol), error=function(e) NULL)
}) %>% bind_rows()

# Clean data
df_sp <- base_table %>% 
  left_join(df_sp, by="Symbol") %>% 
  clean_names("all_caps") %>% 
  select(SYMBOL, SECURITY, WEBSITE, BUSINESS_SUMMARY, everything())

# Save to file
df_sp %>% 
  write_delim(here("01_Data/02_Firms/df_all_sp_firms.txt"), delim = '\t')
