# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, here)
source(here("02_Code/util.R"))

df_startup <- read_delim(here("01_Data/02_Firms/df_startup_firms_en_prox.txt"), delim = '\t')

th <- 0.27
cleantech_fields <- c("Adaption", "Battery", "Biofuels", "CCS", "E-Efficiency", "E-Mobility", "Generation", "Grid", "Materials", "Water")              

df_startup <- df_startup %>% 
  mutate(across(cleantech_fields, function(x) ifelse(x > th, 1, 0)))

df_startup %>% 
  select(cleantech_fields) %>% colSums()

model <- lm(formula = N_EMPLOYEES ~ , data = df_startup[!is.na(df_startup$VC_SEARCH),])
