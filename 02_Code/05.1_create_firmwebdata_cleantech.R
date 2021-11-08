# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, rvest, janitor, memoise, here, openxlsx)


# Scrape base table of companies in 2021 Global Cleantech 100 List
scrape_tab <- function(url){
  
  html <- read_html(url)
  
  tab_base <- html %>% html_table(header = TRUE, convert = TRUE, na.strings = "") %>% 
    .[[1]] %>% 
    head(100) %>% 
    clean_names("all_caps") %>% 
    select(-c(X, COMPANY)) %>% 
    filter(!str_detect(GEOGRAPHY, 'No results found.'))
  
  tab_href <- html %>% html_element(xpath = "//*[@id='gct-table-data']") %>% 
    html_element("tbody") %>% 
    html_elements("a") %>% 
    html_attr("href") %>% 
    as_tibble() %>%
    rename("HREF"="value") %>% 
    mutate(
      COMPANY = str_remove(HREF, "/company/"),
      HREF = paste0("https://i3connect.com/", HREF)
    )
  
  tab <- tab_base %>% 
    bind_cols(tab_href) %>% 
    select(COMPANY, everything())
  
  return(tab)
  
}


# Auxiliary functions:
# Simple string cleaner (1)
clean_pattern <- function(string){
  return(gsub(pattern = '"|:', '', string))
}

# Simple string cleaner (2)
clean_entry <- function(string){
  return(gsub(pattern = '\"|\",', '', string))
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
    entry_list[[clean_pattern(pattern)]] <- clean_entry(entry)
  }
  
  return(entry_list)
}

# Scrape detailed firm info
scrape_firminfo <- function(url, pattern_vector){
  text <- read_html(url) %>%
    html_text2()

  start <- str_locate_all(string = text, pattern = regex('\"company\":'))[[1]][1]
  end <- str_locate_all(string = text, pattern = regex(',\"follow\":'))[[1]][1]
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
# Create empty tibble for final results
df_cleantech <- tibble()

# The Cleantech-100 list exists for all years from 2009-2021. Adding the year to the base url
# leads to the respective year list
for (year in (2017:2021)){
  # First, scrape the base table 
  url <- "https://i3connect.com/gct100/the-list/"
  base_table <- scrape_tab(paste0(url, year))

  # Second, given the HREF in the base table scrape the detailed firm information
  pattern_vector <- c(
    '\"id\":',
    '\"name\":',
    '\"url\":',
    '\"logo_id\":',
    '\"short_description\":',
    '\"website\":',
    '\"year_founded\":',
    '\"telephone\":',
    '\"address\":',
    '\"city\":',
    '\"state\":',
    '\"zip\":',
    '\"country_id\":',
    '\"company_status_id\":',
    '\"num_employees\":',
    '\"ticker_symbol\":',
    '\"overview\":',
    '\"products_description\":',
    '\"development_stage_id\":',
    '\"primary_tag\":',
    '\"total_investment\":',
    '\"updated_at\":',
    '\"investor_status_id\":',
    '\"assets\":',
    '\"long_description\":',
    '\"investment_focus\":',
    '\"key_funds\":',
    '\"investor_type_id\":',
    '\"updated_by_user_id\":',
    '\"company_logo\":',
    '\"standard\":',
    '\"logo_url\":',
    '\"profile_type_id\":',
    '\"hidden\":',
    '\"industry_type_id\":',
    '\"revenue_range_id\":',
    '\"snapshot\":',
    '\"offer\":',
    '\"industry_group\":',
    '\"company_type\":',
    '\"industry_type\":',
    '\"country\":',
    '\"investor_type_name\":',
    '\"stage\":',
    '\"primary_tag_id\":',
    '\"industry_group_id\":')


  df_temp <- lapply(base_table$HREF, function(url) cached_scrape_firminfo(url, pattern_vector)) %>% bind_rows()

  # Clean data
  df_temp <- df_temp %>% 
    na_if('null') %>% 
    na_if("") %>% 
    na_if("N/A") %>% 
    mutate(primary_tag = str_replace(primary_tag, pattern = "\\\\u0026", replacement = "&"),
           industry_group = str_replace(industry_group, pattern = "\\\\u0026", replacement = "&")) %>% 
    clean_names("all_caps") %>% 
    type_convert()
  
  df_cleantech <- df_cleantech %>% 
    bind_rows(df_temp)
  
  print(year)
  
}


# Save to file
df_cleantech %>% 
  distinct() %>% # drop duplicates
  write_delim(here("01_Data/02_Firms/df_cleantech_firms.txt"), delim = '\t')

# Save for labeling
df_cleantech %>% 
  distinct() %>% 
  select(ID, WEBSITE, PRIMARY_TAG, SHORT_DESCRIPTION, OVERVIEW, PRODUCTS_DESCRIPTION) %>% 
  write.xlsx(here("01_Data/02_Firms/df_cleantech_firms_label.xlsx"))
