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

# Scrape website and further info
html <- read_html("https://i3connect.com/company/75f")
firm_info <- html %>% 
  html_element(xpath = "//div[@class='cp-profile-details']")

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
  "//input[@id='edit-street-address']"
  //*[@id="left-tile"]/form/div[2]/div[2]/div
  #left-tile > form > div.tile-content.overview-content > div.cp-profile-right > div
  /html/body/div[1]/div[2]/div/div/div[20]/div[1]/form/div[2]/div[2]/div
  //*[@id="left-tile"]/form/div[2]/div[2]
}

