# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse)




# Colors ------------------------------------------------------------------
bb_blue_dark <- rgb(0, 69, 125, maxColorValue = 255)
bb_blue_medium <- rgb(102, 144, 177, maxColorValue = 255)
bb_blue_light <- rgb(204, 218, 229, maxColorValue = 255)

bb_red_dark <- rgb(230, 68, 79, maxColorValue = 255)
bb_red_medium <- rgb(235, 105, 114, maxColorValue = 255)
bb_red_light <- rgb(240, 143, 149, maxColorValue = 255)

bb_green_dark <- rgb(151, 191, 13, maxColorValue = 255)
bb_green_medium <- rgb(172, 204, 61, maxColorValue = 255)
bb_green_light <- rgb(193, 216, 110, maxColorValue = 255)

ml_green_dark <- "seagreen4"
ml_green_medium <- "seagreen3"
ml_green_light <- "seagreen2"
# ml_green_dark <- "aquamarine4"
# ml_green_medium <- "aquamarine3"
# ml_green_light <- "aquamarine2"

function_gradient_blue <- colorRampPalette(c(bb_blue_light, bb_blue_dark))
function_gradient_green <- colorRampPalette(c(ml_green_light, ml_green_dark))
function_gradient_redTOgreen <- colorRampPalette(c(bb_red_dark, ml_green_dark))
function_gradient_redTOwhiteTOgreen <- colorRampPalette(c(bb_red_dark, "white", ml_green_dark))
function_gradient_redTOblueTOgreen <- colorRampPalette(c(bb_red_dark, bb_blue_dark, ml_green_dark))

# Define some more colors
green <- rgb(0.21545773, 0.43364693, 0.25936727, maxColorValue = 1)
grey <- rgb(119,119,119, alpha = 150, maxColorValue = 255)
greyL <- rgb(159,159,159, maxColorValue = 255)
red <- rgb(178,34,34,alpha = 150,  maxColorValue = 255)
redL <- rgb(226,104,104,maxColorValue = 255)
redL2 <- rgb(240,180,180,maxColorValue = 255)
gold <- rgb(255,215,0, alpha = 150, maxColorValue = 255)
goldL <- rgb(254,227,76, maxColorValue = 255)


# Theme -------------------------------------------------------------------


theme_jod <- theme(
  panel.background = element_blank(),
  panel.grid.major.y = element_line(color = "grey90", size = 0.5),
  panel.grid.minor.y = element_line(color = "grey90", size = 0.5),
  panel.grid.major.x = element_blank(),
  panel.grid.minor.x = element_blank(),
  #panel.grid.major.x = element_blank(),
  #panel.border = element_rect(fill = NA, color = "grey20"),
  #axis.text.x = element_text(family = "Arial", angle = 45, hjust = 1),
  axis.text.x = element_text(color = "black", size = 8),
  axis.text.y = element_text(color = "black", size = 8),
  #axis.title = element_text(size = 12, face = "bold", margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt")),  # not working?
  axis.title.y = element_text(size = 9, face = "bold", vjust = 3,  margin = margin(t = 0, r = 0, b = 0, l = 5, unit = "pt")),
  axis.title.x = element_text(size = 9, face = "bold", vjust = -.1),
  plot.title = element_text(size = 12, hjust = 0),
  legend.text = element_text(size = 9, hjust = 0),
  legend.key = element_blank(),
  axis.ticks.x = element_line(),    # Change x axis ticks only
  axis.ticks.y = element_line(),    # Change y axis ticks only
  #axis.ticks = element_blank(),
  strip.text.x = element_text(size = 9, color = "white", face = "bold"),    # changes facet labels
  strip.text.y = element_text(size = 9, color = "white", face = "bold"),
  #strip.background = element_rect(color="black", fill="grey", size=1, linetype="solid")
  strip.background = element_rect(fill="grey"),
  #plot.margin = unit(c(1,1,1,1), "cm")
)

# Specify geom to update, and list attibutes you want to change appearance of
update_geom_defaults("line", list(size = 1))



# Functions ---------------------------------------------------------------
## Function that joins if missing =========================================
coalesce_join <- function(x, y, by = NULL, suffix = c(".x", ".y"), join = dplyr::left_join, ...) {
  joined <- join(x, y, by = by, suffix = suffix, ...)
  # names of desired output
  cols <- union(names(x), names(y))
  
  to_coalesce <- names(joined)[!names(joined) %in% cols]
  suffix_used <- suffix[ifelse(endsWith(to_coalesce, suffix[1]), 1, 2)]
  # remove suffixes and deduplicate
  to_coalesce <- unique(substr(
    to_coalesce, 
    1, 
    nchar(to_coalesce) - nchar(suffix_used)
  ))
  
  coalesced <- purrr::map_dfc(to_coalesce, ~dplyr::coalesce(
    joined[[paste0(.x, suffix[1])]], 
    joined[[paste0(.x, suffix[2])]]
  ))
  names(coalesced) <- to_coalesce
  
  dplyr::bind_cols(joined, coalesced)[cols]
}




## Function to plot table within pipe =====================================
tab <- function(.data, var){
  dtype <- .data %>% 
    dplyr::select({{var}}) %>% 
    as_vector() %>% 
    class()
  
  .data %>% 
    dplyr::select({{var}}) %>% 
    table(useNA = 'always') %>% 
    as_tibble() %>% 
    rename(!!quo(!!ensym(var)) := '.') %>% 
    mutate(p = round(n/sum(n), 5)) %>% 
    arrange(desc(p)) %>% 
    {if(dtype == 'numeric') mutate_all(., as.numeric) else .}
}
        
        
        
# Setup cleantech scraping --------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, rvest, janitor, memoise, here)


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


# String patterns for detailed firm information
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
  '\"industry_group_id\":'
)