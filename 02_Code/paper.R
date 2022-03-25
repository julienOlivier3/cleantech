# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, here)
source(here("02_Code/util.R"))

df_startup <- read_delim(here("01_Data/02_Firms/df_gp_en_prox.txt"), delim = '\t')

th <- 0.27
cleantech_fields <- c("Adaption", "Battery", "Biofuels", "CCS", "E-Efficiency", "E-Mobility", "Generation", "Grid", "Materials", "Water")              

# Create technology column
df_tech <- df_startup %>% 
  select(crefo, cleantech_fields) %>% 
  pivot_longer(cleantech_fields, names_to = "tech", values_to = "tech_prox") %>% 
  group_by(crefo) %>% 
  filter(tech_prox == max(tech_prox)) %>% 
  ungroup() %>% 
  mutate(tech = ifelse(tech_prox < th, "non-cleantech", tech)) %>% 
  distinct()


df_startup <- df_startup %>% 
  left_join(df_tech, by = "crefo")



df_startup <- df_startup %>% 
  mutate(across(cleantech_fields, function(x) ifelse(x > th, 1, 0)))

df_startup %>% 
  select(cleantech_fields) %>% colSums()

model <- lm(formula = umwelt_wirk ~ `Adaption` + `Battery` + `Biofuels` + `CCS` + `E-Efficiency` + `E-Mobility` + `Generation` + `Grid` + `Materials` + `Water`, data = df_startup)
summary(model)

model_anova <- aov(formula = umwelt_wirk ~ tech, data = df_startup)
summary(model_anova)

model_anova_levels <- TukeyHSD(model_anova, ordered = TRUE)
model_anova_levels <- rownames_to_column(as.data.frame(model_anova_levels$tech)) %>% as_tibble()

model_anova_levels %>% 
  filter(str_detect(rowname, 'non-cleantech'))

plot(model_anova_levels, las = 1)
