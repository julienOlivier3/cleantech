# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, rvest, janitor)


url <- "https://i3connect.com/gct100/the-list"

# Scrape base table of companies in 2021 Global Cleantech 100 List
scrape_tab <- function(url){

  html <- read_html(url)
  
  tab_base <- html %>% html_table(header = TRUE, convert = TRUE, na.strings = "") %>% 
    .[[1]] %>% 
    head(100) %>% 
    clean_names("all_caps") %>% 
    select(-c(X, COMPANY)) 
  
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

base_table <- scrape_tab(url)

# Scrape website and further info
html <- read_html("https://i3connect.com//company/agriprotein")

text <- html %>% html_text2()

start <- str_locate_all(string = text, pattern = regex('\"company\":'))[[1]][1]
end <- str_locate_all(string = text, pattern = regex(',\"follow\":'))[[1]][1]
company_text <- substr(text, start, end-1)

clean_pattern <- function(string){
  return(gsub(pattern = '"|:', '', string))
}

clean_entry <- function(string){
  return(gsub(pattern = '\"', '', string))
}

extract_entries <- function(string, pattern_vector){
  entry_list <- list()
  n_patterns <- length(pattern_vector)
  for (i in 1:(n_patterns-1)){
    pattern <- pattern_vector[i]
    start <- str_locate_all(string, pattern = regex(pattern))[[1]][2]
    end <- str_locate_all(string, pattern = regex(pattern_vector[i+1]))[[1]][1]
    entry <- substr(string, start+1, end-3)
    entry_list[[clean_pattern(pattern)]] <- clean_entry(entry)
  }
  
  return(entry_list)
}

pattern_vector <- c(
  '\"id\":',
  '\"name\":',
  '\"url\":',
  '\"logo_id\":',
  '\"short_description\":',
  '\"website\":',
  '\"year_founded\":'
  
)

extract_entries(company_text, pattern_vector)

firm_info <- html %>% 
  html_elements(xpath = "//div[@class='cp-profile-details']")

class_names <- c("short-description", "company-type", "sector", "stage", "address ", "parent-company", "website")  

for (info in class_names){
  firm_info %>% 
    html_element(xpath = paste0("//div[@class='", info,"']")) %>% 
    html_element("h1") %>% 
    html_text() %>% 
    print()
} 

html %>% 
  html_text()



base_table <- scrape_tab(url)

scrape_firm_info <- function(url){

}

