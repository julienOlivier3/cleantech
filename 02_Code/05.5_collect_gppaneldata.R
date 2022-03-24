# Setup -------------------------------------------------------------------
setwd("Q:\\Meine Bibliotheken\\Research\\Green_startups")

# Read packages -----------------------------------------------------------
library("tidyverse")
library("haven")



# Read data ---------------------------------------------------------------

df_raw <- read_delim(file = file.path(getwd(), "02_Data", "02_Gründungspanel", "gp18_labels", "df_GP18_sustain_question.txt"),
                     delim = "\t")

df_text <- read_delim("Z:\\Archiv-IOEK\\FDZ\\HendrikHansmeier\\texte_minwelle_neu3.txt")

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


df_gp <- read_dta(file = file.path("T:\\2018\\Datenaufbereitung\\!Gesamtdaten\\gpges_kern_w1-w11.dta"))





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
