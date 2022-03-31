# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, here, haven)
source(here("02_Code/util.R"))


# Clean data ---------------------------------------------------------------

df_raw <- read_delim(file = file.path(getwd(), "02_Data", "02_Gründungspanel", "gp18_labels", "df_GP18_sustain_question.txt"),
                     delim = "\t")

df_text <- read_delim("Z:\\Archiv-IOEK\\FDZ\\HendrikHansmeier\\texte_minwelle_neu3.txt")
df_text <- df_text %>% rename(c("w" = "firstwavetext")) %>% 
  mutate(text_jahr = case_when(w == 59 ~ 2020,
                               w == 58 ~ 2020,
                               w == 57 ~ 2019,
                               w == 56 ~ 2019,
                               w == 55 ~ 2018,
                               w == 54 ~ 2018,
                               w == 53 ~ 2017,
                               w == 52 ~ 2017,
                               w == 51 ~ 2016,
                               w == 50 ~ 2016,
                               w == 49 ~ 2015,
                               w == 48 ~ 2015,
                               w == 47 ~ 2014,
                               w == 46 ~ 2014,
                               w == 45 ~ 2013,
                               w == 44 ~ 2013,
                               w == 43 ~ 2012,
                               w == 42 ~ 2012,
                               w == 41 ~ 2011,
                               w == 40 ~ 2011,
                               w == 39 ~ 2010,
                               w == 38 ~ 2010,
                               w == 37 ~ 2009,
                               w == 36 ~ 2009,
                               w == 35 ~ 2008,
                               w == 34 ~ 2008,
                               w == 33 ~ 2007,
                               w == 32 ~ 2007,
                               w == 31 ~ 2006,
                               w == 30 ~ 2006,
                               w == 29 ~ 2005,
                               w == 28 ~ 2005,
                               w == 27 ~ 2004,
                               w == 26 ~ 2004,
                               w == 25 ~ 2003,
                               w == 24 ~ 2003,
                               w == 23 ~ 2002,
                               w == 22 ~ 2002,
                               w == 21 ~ 2001,
                               w == 20 ~ 2001,
                               w == 19 ~ 2000,
                               w == 18 ~ 2000,
                               w == 17 ~ 1999,
                               w == 16 ~ 1999,
                               w == 15 ~ 1998,
                               w == 14 ~ 1998,
                               w == 13 ~ 1997,
                               w == 12 ~ 1997,
                               w == 11 ~ 1996,
                               w == 10 ~ 1996,
                               w == 9 ~ 1995,
                               w == 8 ~ 1995,
                               w == 7 ~ 1994,
                               w == 6 ~ 1994,
                               w == 5 ~ 1993,
                               w == 4 ~ 1993,
                               w == 3 ~ 1992,
                               w == 2 ~ 1992,
                               w == 1 ~ 1991))


df_text %>% 
  filter(gruend_jahr>text_jahr)

df_raw <- df_raw %>% left_join(df_text)
df_raw %>% 
  tab(textyn)

df_raw <- df_raw %>% 
  mutate(text = str_remove(text, "Eingetragener Gegenstand: "))

df_raw <- df_raw %>% 
  mutate(text = str_remove(text, regex("Rechtsform: Freie Berufe   Gründung: \\d\\d\\.\\d\\d\\.\\d\\d\\d\\d"))) %>% 
  mutate(text = str_remove(text, regex("Rechtsform: Freie Berufe   Gründung: \\d\\d\\.\\d\\d\\d\\d"))) %>% 
  mutate(text = str_remove(text, regex("Rechtsform: Freie Berufe   Gründung: \\d\\d\\d\\d"))) %>% 
  mutate(text = str_remove(text, regex("Rechtsform: Freie Berufe"))) 


df_raw <- df_raw %>% 
  mutate(text = str_remove(text, regex("Partnerregister.{1,}am \\d\\d\\.\\d\\d\\.\\d\\d\\d\\d"))) 

df_raw <- df_raw %>% 
  mutate(text = ifelse(str_detect(text, "A L L G E M E I N B E R I C H T"), NA, text)) %>%
  mutate(text = ifelse(str_detect(text, "F E H L B E R I C H T"), NA, text)) 

df_raw <- df_raw %>% 
  mutate(text = str_squish(text))


df_raw <- df_raw %>% 
  mutate(text = ifelse(nchar(text)<=1, NA, text)) 

df_raw <- df_raw %>% 
  select(!c(textyn, hilf1, hilf2))



df_raw %>% 
  write_delim(here("01_Data/02_Firms/df_gp.txt"), delim = '\t')


# Add further GP variables ------------------------------------------------
df_raw <- read_delim(file = here("01_Data/02_Firms/03_StartupPanel/df_GP18_sustain_question.txt"), delim = "\t")
df_startup <- read_delim(here("01_Data/02_Firms/df_gp_en_prox2.txt"), delim = '\t')
df_startup <- df_raw %>% 
  left_join(df_startup[c("crefo",  "gruend_jahr", "length", "text", "text_en", "web", "Adaption", "Battery", "Biofuels", "CCS", "E-Efficiency", "E-Mobility", "Generation", "Grid", "Materials", "Water" )])

df_gp <- read_dta(file = file.path("T:\\2018\\Datenaufbereitung\\!Gesamtdaten\\gpges_kern_w1-w11.dta"))
df_gp_spez <- read_dta(file = file.path("T:\\2018\\Datenaufbereitung\\!Gesamtdaten\\gpges_spez_w11.dta"))

df_gp <- df_gp %>% 
  filter(gpkey %in% df_startup$gpkey) %>% 
  filter(jahr == 2017) %>% 
  select(gpkey, jahr, gr_jahr, branche11, team, anzteam, fue, fuep, fuex, fuek, umsj, ums, gewj, gew, kapa, patj, anzpat, contains("foe"), contains("inno"), contains("produkt")) %>% 
  left_join(df_gp_spez %>% select("gpkey", contains("umwelt")))

cleantech_fields <- c("Adaption", "Battery", "Biofuels", "CCS", "E-Efficiency", "E-Mobility", "Generation", "Grid", "Materials", "Water") 

df_all <- df_gp %>% 
  left_join(df_startup %>% select(gpkey, crefo, cleantech_fields, text_en, web))

# Add company websites
df_url <- read_delim("I:\\!Projekte\\BMBF-2021-DynTOBI\\Daten\\url\\final\\url_panel.csv", delim="\t")

df_all <- df_all %>% 
  left_join(df_url[c("crefo", "year", "url")], by = c("crefo" = "crefo", "gr_jahr" = "year")) 


saveRDS(df_all, file=here("01_Data/02_Firms/03_StartupPanel/df_gp.rds"))

# Some quality checks -----------------------------------------------------

# Missing values
df_raw %>% 
  select(crefo, contains("umwelt_wirk_")) %>% 
  map_dfr(function(x) sum(is.na(x))/length(x))

# Crefos for missings (identifier)
id <- df_raw %>%
  mutate(crefo = as.character(crefo),
         gpkey = as.character(gpkey)) %>% 
  select(crefo, contains("umwelt_wirk_")) %>% 
  mutate(n_missings = rowSums(is.na(.))) %>% 
  filter(n_missings > 0) %>% 
  select(crefo) %>% 
  as_vector()


# Prepare data concerning sustainability question -------------------------
df_sustain <- df_raw %>% 
  # select only variables related to environmental sustainability
  select(crefo, gpkey, jahr, exit, exit_datum, contains("umwelt")) %>% 
  # select only variables related to environmentally sustainable effects for customers
  select(crefo, gpkey, jahr, exit, exit_datum, contains("umwelt_wirk")) %>% 
  # identifiers as character
  mutate(crefo = as.character(crefo),
         gpkey = as.character(gpkey)) %>% 
  # recoding of categorical variable
  mutate_at(vars(contains("umwelt_wirk")), function(x) str_replace_all(x, "nein", "0")) %>% 
  mutate_at(vars(contains("umwelt_wirk")), function(x) str_replace_all(x, "ja, gering", "1")) %>% 
  mutate_at(vars(contains("umwelt_wirk")), function(x) str_replace_all(x, "ja, bedeutend", "2")) %>%
  mutate_at(vars(contains("umwelt_wirk")), function(x) as.integer(x)) %>% 
  # add a exit class variable showing relevant and rather irrelevant start-ups
  mutate(exit_class = ifelse(exit == "existiert sicher", "relevant", "irrelevant")) %>% 
  # calculate overall environmentally sustainable effects for customers
  mutate(umwelt_wirk = rowSums(select(., contains("umwelt_wirk")))) %>% 
  # reorder columns
  select(crefo, gpkey, jahr, exit, exit_class, exit_datum, everything())


# Analyze sustainability question -----------------------------------------

## Visualizations =========================================================


# Barchart by question
df_sustain %>% 
  pivot_longer(cols = str_which(colnames(.), "umwelt_wirk_"), names_to = "var_id", values_to = "sustain_degree") %>% 
  ggplot() +
  geom_bar(aes(x = sustain_degree), na.rm = TRUE) +
  #scale_x_discrete(na.translate = FALSE) +
  facet_wrap(~var_id)

# Barchart overall score
df_sustain %>% 
  ggplot() +
  geom_bar(aes(x = umwelt_wirk))



# Statistics ==============================================================


# Counts overall score
df_sustain %>% 
  #filter(exit == "existiert sicher") %>% 
  group_by(umwelt_wirk, exit_class) %>% 
  summarise(n_obs = n()) %>% 
  view()


# Write results  -----------------------------------------------------------

df_GP18_sustain_question <- df_sustain
write_delim(df_GP18_sustain_question,
            path = file.path(getwd(), "02_Data", "gp18_labels", "df_GP18_sustain_question.txt", fsep = "/"),
            delim = "\t")
