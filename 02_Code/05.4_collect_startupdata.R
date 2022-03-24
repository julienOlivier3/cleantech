# Setup -------------------------------------------------------------------

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, rvest, janitor, memoise, here, rjson, pbapply, stringr, RSelenium, lubridate)
source(here("02_Code/util.R"))

# Auxiliary functions:
# Simple string cleaner
clean_pattern <- function(string){
  string <- gsub(pattern = '\n|\t|\r', ' ', string)
  string <- str_squish(string)
  return(string)
}

# Extraction of firm information for single firm
extract_single_firm <- function(node){
  
  firm_list <- list()
  
  firm_list["HREF"] <- node %>% 
    html_node(css = "div.col-sm-6.col-md-8") %>% 
    html_node("a") %>% 
    html_attr("href")
  
  firm_list["NAME"] <- node %>% 
    html_node(css = "div.col-sm-6.col-md-8") %>% 
    html_text() %>% 
    clean_pattern()
  
  firm_list["SHORT_DESCRIPTION"] <- node %>% 
    html_node(css = "div.col-sm-6.col-md-3") %>% 
    html_text() %>% 
    clean_pattern()
  
  firm_list["WEBSITE"] <- node %>% 
    html_nodes(css = "div.col-md-3") %>% 
    .[2] %>% 
    html_node("a") %>% 
    html_attr("href")
  
  return(firm_list)
  
}

# Get maximum number of pages
get_max <- function(url="https://start-green.net/netzwerk/liste/?oid=green-startup"){
  
  html_full <- read_html(url)
  
  Sys.sleep(1)
  
  loop_end <- html_full %>%
    html_nodes(css = "ul.pagination li") %>%
    .[5] %>%
    html_text() %>%
    as.numeric()
  
  return(loop_end)
}


# Extract firm information on page
extract_single_page <- function(n){
  
  # Create url
  url <- paste0("https://start-green.net/netzwerk/liste/?oid=green-startup&page=", as.character(n))
  
  # Read list entries from html page
  list_entries <- read_html(url) %>% html_nodes(css = "div.row.top7")
  
  # Extract information for all firms
  firm_entries <- lapply(list_entries, function(node) extract_single_firm(node)) %>% bind_rows()
  
  # Loop over entries
  return(firm_entries)
}



# Cache function scrape_firminfo
# Define cache_dir inside the function
cache_dir <- here("02_Code/.rcache")
## Create this directory if it doesn't yet exist
if (!dir.exists(cache_dir)) dir.create(cache_dir)

cached_scrape_firminfo <- memoise(extract_single_page, cache = cache_filesystem(cache_dir))


# Scraping ----------------------------------------------------------------
# Scrape base information


df_base <- pblapply(1:get_max(), function(n) cached_scrape_firminfo(n)) %>% bind_rows()

df_base <- df_base %>% 
  mutate(HREF = paste0("https://start-green.net/", HREF))


url <- "https://start-green.net//netzwerk/gruenes-startup/plant-values-beratung-und-coaching-fur-nachhaltigkeit-und-csr/"


# Extraction of firm information for single firm
extract_single_firm_details <- function(url){
  
  firm_list <- list()
  firm_list["HREF"] <- url
  
  firminfo <- read_html(url) %>% 
    html_node(css = "div.col-md-8.top17")
  
  string <- firminfo %>% html_text2() 
  start <- str_locate_all(string, pattern = "Beschreibung\n\n")[[1]][2]
  end <- str_locate_all(substr(string, start+1, stop = nchar(string)), pattern = "\n\n")[[1]][1]
  firm_list["LONG_DESCRIPTION"] <- substr(string, start+1, stop = start+end-1)
  
  temp <- firminfo %>% 
    html_nodes("p")
  
  i <- temp[lapply(temp, function(x) str_detect(x, 'Geschäftsmodell')) %>% as_vector()] %>% html_text() %>% clean_pattern() %>% str_remove('Geschäftsmodell: ')
  firm_list["BUSINESS_MODEL"] <- ifelse(length(i)==0, NA, i)
  i <- temp[lapply(temp, function(x) str_detect(x, 'Gegründet')) %>% as_vector()] %>% html_text() %>% clean_pattern() %>% str_remove('Gegründet: ')
  firm_list["FOUNDING_YEAR"] <- ifelse(length(i)==0, NA, i)
  i <- temp[lapply(temp, function(x) str_detect(x, 'Mitarbeiter')) %>% as_vector()] %>% html_text() %>% clean_pattern() %>% str_remove('Mitarbeiter: ')
  firm_list["N_EMPLOYEES"] <- ifelse(length(i)==0, NA, i)
  i <- temp[lapply(temp, function(x) str_detect(x, 'Unternehmensphase')) %>% as_vector()] %>% html_text() %>% clean_pattern() %>% str_remove('Unternehmensphase: ')
  firm_list["PHASE"] <- ifelse(length(i)==0, NA, i)
  i <- temp[lapply(temp, function(x) str_detect(x, 'Status Kapitalsuche')) %>% as_vector()] %>% html_text() %>% clean_pattern() %>% str_remove('Status Kapitalsuche: ')
  firm_list["VC_SEARCH"] <- ifelse(length(i)==0, NA, i)
  i <- temp[lapply(temp, function(x) str_detect(x, 'Höhe der gesuchten Finanzierung')) %>% as_vector()] %>% html_text() %>% clean_pattern() %>% str_remove('Höhe der gesuchten Finanzierung: ')
  firm_list["VC_AMOUNT"] <- ifelse(length(i)==0, NA, i)

  return(firm_list)
  
}

url <- "https://start-green.net//netzwerk/gruenes-startup/mybetterworld-gmbh/"
url <- "https://start-green.net//netzwerk/gruenes-startup/binee/"
firminfo <- read_html(url) %>% 
  html_node(css = "div.col-md-8.top17")


cached_scrape_firminfo_details <- memoise(extract_single_firm_details, cache = cache_filesystem(cache_dir))

df_details <- pblapply(df_base$HREF, function(url) cached_scrape_firminfo_details(url)) %>% bind_rows()

df_all <- df_details %>% 
  left_join(df_base) %>% 
  select(NAME, HREF, WEBSITE, SHORT_DESCRIPTION, everything())


# Clean data --------------------------------------------------------------

# FOUNDING_YEAR
df_all <- df_all %>% 
  mutate(FOUNDING_YEAR = ifelse(nchar(FOUNDING_YEAR)>4, NA, FOUNDING_YEAR)) %>% 
  mutate(FOUNDING_YEAR = ifelse(FOUNDING_YEAR==1, NA, FOUNDING_YEAR)) %>% 
  mutate(FOUNDING_YEAR = as.integer(FOUNDING_YEAR)) 

# WEBSITE
df_all <- df_all %>% 
  mutate(WEBSITE = ifelse(WEBSITE=="#map", NA, WEBSITE)) %>% 
  mutate(WEBSITE = ifelse(str_detect(WEBSITE, "tel:"), NA, WEBSITE))

# BUSINESS_MODEL
df_all <- df_all %>% 
  mutate(BUSINESS_MODEL = ifelse(nchar(BUSINESS_MODEL)>3, NA, BUSINESS_MODEL)) %>% 
  mutate(BUSINESS_MODEL = ifelse(BUSINESS_MODEL=="", NA, BUSINESS_MODEL))

# N_EMPLOYEES
df_all <- df_all %>% 
  mutate(N_EMPLOYEES = ifelse(nchar(N_EMPLOYEES)>3, NA, N_EMPLOYEES)) %>% 
  mutate(N_EMPLOYEES = str_remove(N_EMPLOYEES, "-")) %>% 
  mutate(N_EMPLOYEES = as.integer(N_EMPLOYEES)) 

# SHORT_DESCRIPTION
df_all <- df_all %>%
  mutate(SHORT_DESCRIPTION = ifelse(is.na(LONG_DESCRIPTION) & !is.na(SHORT_DESCRIPTION), NA, SHORT_DESCRIPTION)) 

# LONG_DESCRIPTION
df_all <- df_all %>% 
  mutate(LONG_DESCRIPTION = ifelse(nchar(SHORT_DESCRIPTION) > nchar(LONG_DESCRIPTION), SHORT_DESCRIPTION, LONG_DESCRIPTION)) 



# Write data --------------------------------------------------------------

# Save to file
df_all %>% 
  write_delim(here("01_Data/02_Firms/df_startup_firms.txt"), delim = '\t')

df_temp <- read_delim(here("01_Data/02_Firms/df_startup_firms.txt"), delim = '\t')
