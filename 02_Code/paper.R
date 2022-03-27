# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(here, labelled, hablar, MASS, tidyverse, VIM, Hmisc, psych)
source(here("02_Code/util.R"))



# Read data ---------------------------------------------------------------

df_startup <- readRDS(here("01_Data/02_Firms/03_StartupPanel/df_gp.rds"))
df_startup <- df_startup %>% rename(c("E_Efficiency" = "E-Efficiency", "E_Mobility" = "E-Mobility"))


th <- 0.27
cleantech_fields <- c("Adaption", "Battery", "Biofuels", "CCS", "E_Efficiency", "E_Mobility", "Generation", "Grid", "Materials", "Water")              

# Select relevant columns for regression
df_model <- df_startup %>% 
  mutate(age = 2018 - gr_jahr) %>% 
  mutate_at(vars(contains("branche11")), function(x) haven::as_factor(x, levels="label")) %>% 
  mutate_at(vars(contains("produkt")), function(x) haven::as_factor(x, levels="label")) %>% 
  mutate_at(vars(contains("umwelt")), function(x) haven::as_factor(x, levels="label")) %>% 
  mutate_at(vars(contains("umwelt")), function(x) str_replace_all(x, "nein", "0")) %>% 
  mutate_at(vars(contains("umwelt")), function(x) str_replace_all(x, "ja, gering", "1")) %>% 
  mutate_at(vars(contains("umwelt")), function(x) str_replace_all(x, "ja, bedeutend", "2")) %>%
  mutate_at(vars(contains("umwelt")), function(x) as.integer(x)) %>% 
  # Categorical dependent variables
  rowwise() %>%
  mutate(umwelt_wirk_cat = hablar::max_(c_across(contains("umwelt_wirk")), ignore_na = TRUE),
         umwelt_inno_cat = hablar::max_(c_across(contains("umwelt_inno")), ignore_na = TRUE)) %>% 
  ungroup() %>% 
  mutate(
    umwelt_wirk_cat = case_when(
      umwelt_wirk_cat == 0 ~ "nein",
      umwelt_wirk_cat == 1 ~ "ja, gering",
      umwelt_wirk_cat == 2 ~ "ja, bedeutend"),
    umwelt_wirk_cat = factor(umwelt_wirk_cat, levels = c("nein", "ja, gering", "ja, bedeutend"), ordered = TRUE),
    umwelt_inno_cat = case_when(
      umwelt_inno_cat == 0 ~ "nein",
      umwelt_inno_cat == 1 ~ "ja, gering",
      umwelt_inno_cat == 2 ~ "ja, bedeutend"),
    umwelt_inno_cat = factor(umwelt_inno_cat, levels = c("nein", "ja, gering", "ja, bedeutend"), ordered = TRUE)
  ) %>%
  # Interval dependent variables
  rowwise() %>% 
  mutate(umwelt_wirk = sum(c_across(umwelt_wirk_gesverbr:umwelt_wirk_haltbar), na.rm = FALSE),
         umwelt_inno = sum(c_across(umwelt_inno_gesverbr:umwelt_inno_haltbar), na.rm = FALSE)) %>% 
  ungroup() %>% 
  select(gpkey, age, produkt_kat, branche11, team, anzteam, fue, fuep, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, contains("umwelt_wirk"), contains("umwelt_inno"), cleantech_fields, url, text_en)

# Create technology column
df_tech <- df_startup %>% 
  select(gpkey, cleantech_fields) %>% 
  pivot_longer(cleantech_fields, names_to = "tech", values_to = "tech_prox") %>% 
  group_by(gpkey) %>% 
  filter(tech_prox == max(tech_prox)) %>% 
  ungroup() %>% 
  mutate(tech = ifelse(tech_prox > th, tech, "non-cleantech")) %>% 
  mutate(tech = factor(tech, levels = c('non-cleantech', cleantech_fields))) %>% 
  distinct() %>% 
  mutate(cleantech = ifelse(tech_prox > th, 1, 0))
  
df_model <- df_model %>% 
  left_join(df_tech, by = "gpkey")

df_model %>% tab(umwelt_wirk)
df_model %>% tab(umwelt_inno)
df_model %>% tab(umwelt_wirk_cat)
df_model %>% tab(umwelt_inno_cat)


df_startup %>% 
  select(gpkey, cleantech_fields) %>% 
  pivot_longer(cleantech_fields, names_to = "tech", values_to = "tech_prox") %>% 
  ggplot() +
  geom_boxplot(aes(x=tech, y=tech_prox))



# Create balanced cross section -------------------------------------------
df_model %>% 
  select(umwelt_wirk_cat, umwelt_inno_cat) %>% 
  rowwise() %>% 
  map(function(x) table(x, useNA = "always"))

df_model %>% 
  select(umwelt_wirk_cat, umwelt_inno_cat) %>% 
  rowwise() %>% 
  map(function(x) sum(!is.na(x)))

# Drop observations w/o response to environmental question
n_all <- nrow(df_model)
df_model <- df_model %>% 
  filter(!is.na(umwelt_wirk_cat) & !is.na(umwelt_inno_cat))
n_1 <- nrow(df_model)
print(paste("Drop", n_all-n_1, "observations w/o response to environmental question"))

# Impute missing descriptions with website info
df_model %>% 
  filter(is.na(cleantech) & !is.na(url)) %>% 
  select(gpkey, url) #%>% # Imputation done!
  #write_csv(here("01_Data/02_Firms/03_StartupPanel/df_no_desc_but_url.csv"))

# Drop observations w/o textual information
df_model <- df_model %>% 
  filter(!is.na(cleantech))
n_2 <- nrow(df_model)
print(paste("Drop", n_1-n_2, "observations w/o textual information"))

df_model %>% 
  map(function(x) sum(is.na(x)))

# Some simple kNN imputation
df_model <- kNN(df_model, 
    metric = "mahalanobis", 
    variable = c("team", "anzteam", "fue", "fuep", "umsj"), 
    dist_var = c("age", "branche11", "team", "anzteam", "fue", "umsj"), 
    k=10, 
    imp_var = FALSE, 
    useImputedDist = FALSE) %>% as_tibble() 

df_model %>% 
  map(function(x) sum(is.na(x)))

saveRDS(df_model, file=here("01_Data/02_Firms/03_StartupPanel/df_gp_impute.rds"))
df_model %>% write_delim(file=here("01_Data/02_Firms/03_StartupPanel/df_gp_impute.txt"), delim="\t")

# Descriptive statistics --------------------------------------------------
desc_list <- vector(mode = "list", length = 2)
names(desc_list) <- c("Hmisc", "psych")

desc_list[[1]] <- apply(df_model %>% select(cleantech, tech_prox, fue, fuep, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, age, team, anzteam, cleantech_fields, contains("umwelt")), MARGIN = 2, function(x) Hmisc::describe(x))
desc_list[[2]] <- sapply(df_model %>% select(cleantech, tech_prox, fue, fuep, fuex, fuek, umsj, ums, gewj, gew, age, team, anzteam, cleantech_fields, 
                           umwelt_wirk_gesverbr:umwelt_wirk_haltbar, umwelt_wirk, umwelt_inno_gesverbr:umwelt_inno_haltbar, umwelt_inno), function(x) psych::describe(x))


saveRDS(desc_list, file=here("01_Data/02_Firms/03_StartupPanel/desc_list.rds"))

# Correlations ------------------------------------------------------------
corr_list <- vector(mode = "list", length = 2)
names(corr_list) <- c("pearson", "spearman")

corr_list[[1]] <- rcorr(df_model %>% select(umwelt_inno, umwelt_wirk, cleantech, tech_prox, fue, fuep, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, age, anzteam, cleantech_fields) %>% as.matrix(), type = "pearson")
corr_list[[2]] <- rcorr(df_model %>% select(umwelt_inno, umwelt_wirk, cleantech, tech_prox, fue, fuep, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, age, anzteam, cleantech_fields) %>% as.matrix(), type = "spearman")

saveRDS(corr_list, file=here("01_Data/02_Firms/03_StartupPanel/corr_list.rds"))

# Regression analyses -----------------------------------------------------


## Ordered Logit ==========================================================

model_equations <- c(
  # Wirkung
  "umwelt_wirk_cat ~ tech_prox",
  "umwelt_wirk_cat ~ tech_prox + branche11",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam",
  "umwelt_wirk_cat ~ tech_prox + branche11 + age",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + age",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + fue",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + fuep",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + scale(fuek)",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + age + fue",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + age + fuep",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + age + scale(fuek)",
  "umwelt_wirk_cat ~ tech_prox + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox + branche11 + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + fue + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox + branche11 + anzteam + age + fue + produkt_kat",
  
  "umwelt_wirk_cat ~ cleantech",
  "umwelt_wirk_cat ~ cleantech + branche11",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam",
  "umwelt_wirk_cat ~ cleantech + branche11 + age",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + age",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + fue",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + fuep",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + scale(fuek)",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + age + fue",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + age + fuep",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + age + scale(fuek)",
  "umwelt_wirk_cat ~ cleantech + produkt_kat",
  "umwelt_wirk_cat ~ cleantech + branche11 + produkt_kat",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + produkt_kat",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + fue + produkt_kat",
  "umwelt_wirk_cat ~ cleantech + branche11 + anzteam + age + fue + produkt_kat",
  
  "umwelt_wirk_cat ~ tech_prox*fue",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + age",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + age",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + fue",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + fuep",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + scale(fuek)",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + age + fue",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + age + fuep",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + age + scale(fuek)",
  "umwelt_wirk_cat ~ tech_prox*fue + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + fue + produkt_kat",
  "umwelt_wirk_cat ~ tech_prox*fue + branche11 + anzteam + age + fue + produkt_kat",
  
  "umwelt_wirk_cat ~ cleantech*fue",
  "umwelt_wirk_cat ~ cleantech*fue + branche11",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + age",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + age",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + fue",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + fuep",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + scale(fuek)",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + age + fue",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + age + fuep",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + age + scale(fuek)",
  "umwelt_wirk_cat ~ cleantech*fue + produkt_kat",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + produkt_kat",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + produkt_kat",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + fue + produkt_kat",
  "umwelt_wirk_cat ~ cleantech*fue + branche11 + anzteam + age + fue + produkt_kat",
  
  # Innovation
  "umwelt_inno_cat ~ tech_prox",
  "umwelt_inno_cat ~ tech_prox + branche11",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam",
  "umwelt_inno_cat ~ tech_prox + branche11 + age",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + age",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + fue",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + fuep",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + scale(fuek)",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + age + fue",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + age + fuep",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + age + scale(fuek)",
  "umwelt_inno_cat ~ tech_prox + produkt_kat",
  "umwelt_inno_cat ~ tech_prox + branche11 + produkt_kat",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + produkt_kat",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + fue + produkt_kat",
  "umwelt_inno_cat ~ tech_prox + branche11 + anzteam + age + fue + produkt_kat",
  
  "umwelt_inno_cat ~ cleantech",
  "umwelt_inno_cat ~ cleantech + branche11",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam",
  "umwelt_inno_cat ~ cleantech + branche11 + age",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + age",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + fue",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + fuep",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + scale(fuek)",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + age + fue",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + age + fuep",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + age + scale(fuek)",
  "umwelt_inno_cat ~ cleantech + produkt_kat",
  "umwelt_inno_cat ~ cleantech + branche11 + produkt_kat",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + produkt_kat",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + fue + produkt_kat",
  "umwelt_inno_cat ~ cleantech + branche11 + anzteam + age + fue + produkt_kat",
  
  "umwelt_inno_cat ~ tech_prox*fue",
  "umwelt_inno_cat ~ tech_prox*fue + branche11",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + age",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + age",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + fue",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + fuep",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + scale(fuek)",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + age + fue",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + age + fuep",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + age + scale(fuek)",
  "umwelt_inno_cat ~ tech_prox*fue + produkt_kat",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + produkt_kat",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + produkt_kat",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + fue + produkt_kat",
  "umwelt_inno_cat ~ tech_prox*fue + branche11 + anzteam + age + fue + produkt_kat",
  
  "umwelt_inno_cat ~ cleantech*fue",
  "umwelt_inno_cat ~ cleantech*fue + branche11",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + age",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + age",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + fue",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + fuep",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + scale(fuek)",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + age + fue",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + age + fuep",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + age + scale(fuek)",
  "umwelt_inno_cat ~ cleantech*fue + produkt_kat",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + produkt_kat",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + produkt_kat",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + fue + produkt_kat",
  "umwelt_inno_cat ~ cleantech*fue + branche11 + anzteam + age + fue + produkt_kat"
  
)

model_list <- vector(mode = "list", length = length(model_equations))
names(model_list) <- model_equations
result_table <- tibble(
  model = character(),
  odds = numeric(),
  p_value = numeric(),
  N = numeric()
)

for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  print(model_equation)
  
  polr_mod <- polr(formula = as.formula(model_equation), data = df_model, na.action = na.omit)
  model_list[[i]] <- polr_mod
  
  temp <- summary(polr_mod)$coefficients[1,]
  odds <- exp(temp["Value"])
  p_value <- pnorm(abs(temp["t value"]), lower.tail = FALSE) * 2
  N = nobs(polr_mod)
  result_table <- result_table %>% bind_rows(tibble(model = model_equation, odds = odds, p_value = p_value, N = N))
  
}

result_table %>% 
  filter(str_detect(model, "umwelt_inno")) %>% 
  arrange(desc(p_value))

saveRDS(model_list, file=here("01_Data/02_Firms/03_StartupPanel/cat_model_list.rds"))


model_equations <- lapply(cleantech_fields, function(tech) paste("umwelt_wirk_cat ~", tech, "+ fue + branche11 + anzteam + age")) %>% as_vector()

df_model2 <- df_model %>% 
  mutate(across(cleantech_fields, function(x) ifelse(x > th, 1, 0)))


model_list <- vector(mode = "list", length = length(model_equations))
names(model_list) <- model_equations
result_table <- tibble(
  model = character(),
  odds = numeric(),
  p_value = numeric(),
  N = numeric()
)

for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  print(model_equation)
  
  polr_mod <- polr(formula = as.formula(model_equation), data = df_model2, na.action = na.omit)
  model_list[[i]] <- polr_mod
  
  temp <- summary(polr_mod)$coefficients[1,]
  odds <- exp(temp["Value"])
  p_value <- round(pnorm(abs(temp["t value"]), lower.tail = FALSE) * 2, 5)
  N = nobs(polr_mod)
  result_table <- result_table %>% bind_rows(tibble(model = model_equation, odds = odds, p_value = p_value, N = N))
  
}

result_table %>% 
  arrange(desc(p_value))



polr_mod <- readRDS(here("01_Data/02_Firms/03_StartupPanel/cat_model_list.rds"))
cleantech_fields
model_equation <- "umwelt_wirk_cat ~ Adaption + Battery + Biofuels + CCS + E_Efficiency + E_Mobility + Generation + Grid + Materials + Water + fue + branche11 + age + anzteam + produkt_kat"
model_equation <- "umwelt_wirk_cat ~ cleantech + fue + branche11 + age + anzteam"
polr_mod <- polr(formula = as.formula(model_equation), data = df_model, na.action = na.omit)
coef_res <- summary(polr_mod)$coefficients
coef_res <- coef_res %>% as_tibble(rownames = "Vaiable") %>% 
  mutate(Odds = exp(Value),
         p_value = round(pnorm(abs(`t value`), lower.tail = FALSE) * 2,4))
coef_res

## Linear Model ===========================================================

model_equations <- lapply(model_equations, function(x) str_replace(x, pattern = "umwelt_wirk_cat", "umwelt_wirk")) %>% as_vector()
model_equations <- lapply(model_equations, function(x) str_replace(x, pattern = "umwelt_inno_cat", "umwelt_inno")) %>% as_vector()

model_list <- vector(mode = "list", length = length(model_equations))
names(model_list) <- model_equations
result_table <- tibble(
  model = character(),
  est = numeric(),
  p_value = numeric(),
  N = numeric()
)

for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  print(model_equation)
  
  lm_mod <- lm(formula = as.formula(model_equation), data = df_model, na.action = na.omit)
  model_list[[i]] <- lm_mod
  
  temp <- summary(lm_mod)$coefficients[2,]
  est <- temp["Estimate"]
  p_value <- temp["Pr(>|t|)"]
  N <- nobs(lm_mod)
  result_table <- result_table %>% bind_rows(tibble(model = model_equation, est = est, p_value = p_value, N = N))
  
}

saveRDS(model_list, file=here("01_Data/02_Firms/03_StartupPanel/lm_model_list.rds"))

