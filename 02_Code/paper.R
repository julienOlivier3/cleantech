# Setup -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(here, tidyverse, labelled, hablar, MASS,VIM, Hmisc, psych, margins, pscl, xtable)
source(here("02_Code/util.R"))



# Functions ---------------------------------------------------------------

# Function to return publication ready coefficient estimates
clean_digit <- function(digit, ndigits_display=3, ndigits_round=3){
  return(sprintf(paste0("%.", ndigits_display, "f"), round(digit, ndigits_round)))
}

# Function to extract relevant info from ordered Logit model
clean_results <- function(model, ndigits_display=3, ndigits_round=3){
  res <- summary(model)$coefficients %>% 
    as_tibble(rownames = "variable") %>% 
    mutate(
      odds = map_dbl(Value, function(x) exp(x)),
      p_value = map_dbl(`t value`, function(x) round(pnorm(abs(x), lower.tail = FALSE) * 2, 6))) %>% 
    mutate(odds_p = case_when(
      p_value < 0.01 ~ paste0(clean_digit(odds, ndigits_display, ndigits_round), "***"),
      p_value < 0.05 & p_value >= 0.01 ~ paste0(clean_digit(odds, ndigits_display, ndigits_round), "**"),
      p_value < 0.1 & p_value >= 0.05 ~ paste0(clean_digit(odds, ndigits_display, ndigits_round), "*"),
      p_value >= 0.1 ~ paste0(clean_digit(odds, ndigits_display, ndigits_round))
    )) %>% 
    dplyr::select(variable, odds_p) %>% 
    bind_rows(tibble(variable = "N", odds_p = as.character(nobs(model)))) %>% 
    bind_rows(tibble(variable = "BIC", odds_p = as.character(clean_digit(BIC(model), ndigits_display, ndigits_round)))) %>% 
    bind_rows(tibble(variable = "Pseudo_R2", odds_p = as.character(clean_digit(pR2(model)['McFadden'], ndigits_display, ndigits_round)))) 
  
  return(res)
}

# Function to extract relevant info from Logit model
clean_results_logit <- function(model, ndigits_display=3, ndigits_round=3){
  helper <- summary(model)$coefficients %>% as_tibble(rownames = "factor") %>% dplyr::select(factor) %>% filter(!(factor == "(Intercept)")) %>% as_vector()
  
  res <- margins_summary(model) %>% 
    as_tibble() %>% 
    mutate(ame_p = case_when(
      p < 0.01 ~ paste0(clean_digit(AME, ndigits_display, ndigits_round), "***"),
      p < 0.05 & p >= 0.01 ~ paste0(clean_digit(AME, ndigits_display, ndigits_round), "**"),
      p < 0.1 & p >= 0.05 ~ paste0(clean_digit(AME, ndigits_display, ndigits_round), "*"),
      p >= 0.1 ~ paste0(clean_digit(AME, ndigits_display, ndigits_round))
    )) %>% 
    dplyr::select(factor, ame_p) %>% 
    arrange(helper) %>% 
    bind_rows(tibble(factor = "N", ame_p = as.character(nobs(model)))) %>% 
    bind_rows(tibble(factor = "BIC", ame_p = as.character(clean_digit(BIC(model), ndigits_display, ndigits_round)))) %>% 
    bind_rows(tibble(factor = "Pseudo_R2", ame_p = as.character(clean_digit(pR2(model)['McFadden'], ndigits_display, ndigits_round)))) 
  
  return(res)
}


# Function to clean up regression results and to produce nice regression tables
create_reg_tab <- function(df_res, all_sector=TRUE, all_product=FALSE){

  df_bottom <- df_res %>% filter(variable %in% bottom)
  df_top <- df_res %>% filter(!(variable %in% bottom))
  if (all_sector){
    sector_controls <- rep("Y", length(df_top)-1)
  }
  else{
    sector_controls <- rep("N", length(df_top)-1)
  }
  if (all_product){
    product_controls <- rep("Y", length(df_top)-1)
  }
  else{
    product_controls <- c(rep("N", length(df_top)-2), "Y")
  }
  
  
  df_controls <- tibble(`Sector controls` = sector_controls,
                        `Product type controls` = product_controls) %>% 
    t() %>% 
    as_tibble(rownames = "variable")
  colnames(df_controls) <- colnames(df_top) 
  
  
  df_res <- df_top %>% 
    bind_rows(df_controls) %>% 
    bind_rows(df_bottom) 
  
  df_latex <- df_res %>% 
    filter(!str_detect(variable, "branche.{1,}")) %>% 
    filter(!str_detect(variable, "produkt.{1,}")) %>% 
    filter(!str_detect(variable, "ja")) %>% 
    filter(!str_detect(variable, "BIC")) %>% 
    mutate(variable = case_when(
      variable == 'tech_prox' ~ '\\textsc{TechProx}$_{max}$',
      variable == 'tech_prox_t' ~ '\\textsc{TechProx}$_{t}$',
      variable == 'cleantech' ~ '\\textsc{CleanTech}',
      variable == 'cleantech_t' ~ '\\textsc{CleanTech}$_{t}$',
      (variable == 'bes') | (variable == 'log(bes)') ~ 'log(size)',
      variable == 'fue' ~ 'R\\&D',
      variable == 'fuep_bes' ~ 'R\\&D intensity',
      variable == 'umsj' ~ 'returns',
      variable == 'gewj' ~ 'break even',
      variable == 'foe' ~ 'subsidy',
      variable == 'anzteam' ~ 'team size',
      variable == 'uni' ~ 'university',
      variable == 'N' ~ '$N$',
      variable == 'Pseudo_R2' ~ 'Pseudo $R^2$',
      TRUE ~ variable)
    )
  
  return(df_latex)
    
}



# Read data ---------------------------------------------------------------

df_startup <- readRDS(here("01_Data/02_Firms/03_StartupPanel/df_gp.rds"))
#df_startup <- df_startup %>% rename(c("E_Efficiency" = "E-Efficiency", "E_Mobility" = "E-Mobility"))


th <- 0.27
cleantech_fields <- c("Adaption", "Battery", "Biofuels", "CCS", "E_Efficiency", "E_Mobility", "Generation", "Grid", "Materials", "Water")              

# Select relevant columns for regression
df_model <- df_startup %>% 
  mutate(age = 2017 - gr_jahr) %>% 
  mutate(foe = ifelse(rowSums(dplyr::select(., foe_ba, foe_kfw, foe_land, foe_kokr, foe_bueba, foe_bund, foe_eu))>0, 1, 0)) %>% 
  mutate(ums_hilf = ifelse(ums==0, 1, ums)) %>% 
  mutate(bes_hilf = rowSums(dplyr::select(., anzteam, bes_l, bes_m, bes_h), na.rm = TRUE),
         bes_hilf = ifelse(bes_hilf==0, NA, bes_hilf),
         bes = ifelse(bes_hilf < fuep, fuep, bes_hilf)) %>% 
  mutate(fuek_ums = fuek/ums_hilf,
         fuep_bes = fuep/bes) %>% 
  mutate(uni = ifelse(a_uni == 1 | a_dr == 1, 1, 0)) %>% 
  mutate(gewj = ifelse(gewj == 1 | gewj == 3, 1, 0)) %>% 
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
  # Binary dependent variables
  mutate(umwelt_wirk_bin = ifelse(str_detect(umwelt_wirk_cat, "ja"), 1, 0),
         umwelt_inno_bin = ifelse(str_detect(umwelt_inno_cat, "ja"), 1, 0)) %>% 
  # Interval dependent variables
  rowwise() %>% 
  mutate(umwelt_wirk = sum(c_across(umwelt_wirk_gesverbr:umwelt_wirk_haltbar), na.rm = FALSE),
         umwelt_inno = sum(c_across(umwelt_inno_gesverbr:umwelt_inno_haltbar), na.rm = FALSE)) %>% 
  ungroup() %>% 
  dplyr::select(gpkey, age, produkt_kat, branche11, team, anzteam, uni, foe, bes, 
         fue, fuep, fuek_ums, fuep_bes, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, 
         contains("umwelt_wirk"), contains("umwelt_inno"), cleantech_fields, url, web, text_en)

# Create technology column
df_tech <- df_startup %>% 
  dplyr::select(gpkey, cleantech_fields) %>% 
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
  dplyr::select(gpkey, cleantech_fields) %>% 
  pivot_longer(cleantech_fields, names_to = "tech", values_to = "tech_prox") %>% 
  ggplot() +
  geom_boxplot(aes(x=tech, y=tech_prox))



# Create balanced cross section -------------------------------------------
df_model %>% 
  dplyr::select(umwelt_wirk_cat, umwelt_inno_cat) %>% 
  rowwise() %>% 
  map(function(x) table(x, useNA = "always"))

df_model %>% 
  dplyr::select(umwelt_wirk_cat, umwelt_inno_cat) %>% 
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
  dplyr::select(gpkey, url) #%>% # Imputation done!
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
                variable = c("team", "anzteam", "fue", "fuep", "umsj", "foe", "bes", "fuep_bes", "fuek_ums", "uni"), 
                dist_var = c("age", "branche11", "team", "bes", "fue", "umsj"), 
                k=10, 
                imp_var = FALSE, 
                useImputedDist = FALSE) %>% as_tibble() 

df_model %>% 
  map(function(x) sum(is.na(x))) 

#saveRDS(df_model, file=here("01_Data/02_Firms/03_StartupPanel/df_gp_impute.rds"))
#df_model %>% write_delim(file=here("01_Data/02_Firms/03_StartupPanel/df_gp_impute.txt"), delim="\t")

# Descriptive statistics --------------------------------------------------
desc_list <- vector(mode = "list", length = 2)
names(desc_list) <- c("Hmisc", "psych")

#desc_list[[1]] <- apply(df_model %>% select(tech_prox, cleantech, bes, age, foe, fue, fuep_bes, fuek_ums, umsj, ums, gewj, gew, team, anzteam, uni), MARGIN = 2, function(x) Hmisc::describe(x))
desc_list[[2]] <- sapply(df_model %>% dplyr::select(tech_prox, cleantech, bes, age, fue, fuep_bes, umsj, gewj, foe, anzteam, uni), function(x) psych::describe(x))

desc_list$psych %>% 
  as_tibble(rownames = "Variable") %>% 
  mutate_all(function(x) unlist(x)) %>% 
  filter(Variable %in% c("mean", "sd", "min", "max", "n")) %>% 
  pivot_longer(tech_prox:uni) %>% 
  pivot_wider(id_cols = Variable:value, names_from=Variable)

#saveRDS(desc_list, file=here("01_Data/02_Firms/03_StartupPanel/desc_list.rds"))

# Correlations ------------------------------------------------------------
corr_list <- vector(mode = "list", length = 2)
names(corr_list) <- c("pearson", "spearman")

corr_list[[1]] <- rcorr(df_model %>% select(umwelt_inno, umwelt_wirk, cleantech, tech_prox, foe, bes, fuek_ums, fuep_bes, fue, fuep, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, age, anzteam, cleantech_fields) %>% as.matrix(), type = "pearson")
corr_list[[2]] <- rcorr(df_model %>% select(umwelt_inno, umwelt_wirk, cleantech, tech_prox, foe, bes, fuek_ums, fuep_bes, fue, fuep, fuex, fuek, patj, anzpat, umsj, ums, gewj, gew, age, anzteam, cleantech_fields) %>% as.matrix(), type = "spearman")



#saveRDS(corr_list, file=here("01_Data/02_Firms/03_StartupPanel/corr_list.rds"))

# Regression analyses -----------------------------------------------------
## Ordered Logit (full) ===================================================
### Umweltwirkung #########################################################

## TechProx

df_reg <- df_model %>% 
  mutate_at(cleantech_fields, function(x) x*100) 
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(cleantech_fields)){
  tech <- cleantech_fields[i]
  model_equation <- paste("umwelt_wirk_cat ~", tech, "+ branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni + produkt_kat")
  polr_mod <- polr(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, method = "logistic")
  df_temp <- clean_results(model = polr_mod)
  colnames(df_temp)  <- c("variable", tech)
  df_temp <- df_temp %>% mutate(variable = ifelse(variable == tech, "tech_prox_t", variable))
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% left_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res, all_sector = TRUE, all_product = TRUE)
df_latex <- df_latex %>% select(variable:E_Efficiency, Generation:Materials, E_Mobility, Water)

print.xtable(df_latex, type = "latex")


## cleantech 

df_reg <- df_model %>% 
  mutate(across(cleantech_fields, function(x) ifelse(x > th, 1, 0)))
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(cleantech_fields)){
  tech <- cleantech_fields[i]
  model_equation <- paste("umwelt_wirk_cat ~", tech, "+ branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni + produkt_kat")
  polr_mod <- polr(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, method = "logistic")
  df_temp <- clean_results(model = polr_mod)
  colnames(df_temp)  <- c("variable", tech)
  df_temp <- df_temp %>% mutate(variable = ifelse(variable == tech, "cleantech_t", variable))
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% left_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res, all_sector = TRUE, all_product = TRUE)
df_latex <- df_latex %>% select(variable:E_Efficiency, Generation:Materials, E_Mobility, Water)

print.xtable(df_latex, type = "latex")

## Ordered Logit (full) ===================================================
### Umweltwirkung #########################################################
### TechProx

model_equations <- c(  
  "umwelt_wirk_cat ~ tech_prox + branche11",
  #"umwelt_wirk_cat ~ tech_prox + branche11 + log(bes)",
  "umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age",
  #"umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue",
  #"umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes",
  "umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe",
  #"umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj",
  "umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj",
  #"umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam",
  "umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni",
  "umwelt_wirk_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni + produkt_kat"
)

df_reg <- df_model %>% 
  mutate(tech_prox = tech_prox*100) 
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- polr(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, method = "logistic")
  df_temp <- clean_results(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res)
df_latex

print.xtable(df_latex, 
             type = "latex", 
             #include.rownames = FALSE,
             #floating = FALSE, 
             format.args = list(digits = 3, big.mark = " ", decimal.mark = ","))



### cleantech

model_equations <- lapply(model_equations, function(x) str_replace(x, "tech_prox", "cleantech")) %>% as_vector()

df_reg <- df_model
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- polr(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, method = "logistic")
  df_temp <- clean_results(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_bottom <- df_res %>% filter(variable %in% bottom)
df_top <- df_res %>% filter(!(variable %in% bottom))

df_res <- df_top %>% 
  bind_rows(df_bottom)



df_latex <- df_res %>% 
  filter(!str_detect(variable, "branche.{1,}")) %>% 
  filter(!str_detect(variable, "produkt.{1,}")) %>% 
  filter(!str_detect(variable, "ja"))

print.xtable(df_latex, type = "latex")


### Umweltinnovation ######################################################
### TechProx


model_equations <- c(  
  "umwelt_inno_cat ~ tech_prox + branche11",
  "umwelt_inno_cat ~ tech_prox + branche11 + log(bes)",
  #"umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + age",
  #"umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + age + fue",
  #"umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes",
  #"umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe",
  "umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + fue + foe",
  #"umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj",
  "umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + fue + foe + umsj + gewj",
  #"umwelt_inno_cat ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam",
  "umwelt_inno_cat ~ tech_prox + branche11 + log(bes)  + fue  + foe + umsj + gewj + anzteam + uni",
  "umwelt_inno_cat ~ tech_prox + branche11 + log(bes)  + fue  + foe + umsj + gewj + anzteam + uni + produkt_kat"
)

df_reg <- df_model %>% 
  mutate(tech_prox = tech_prox*100) 
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- polr(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, method = "logistic")
  df_temp <- clean_results(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res)
df_latex

print.xtable(df_latex, 
             type = "latex", 
             #include.rownames = FALSE,
             #floating = FALSE, 
             format.args = list(digits = 3, big.mark = " ", decimal.mark = ","))




### cleantech

model_equations <- lapply(model_equations, function(x) str_replace(x, "tech_prox", "cleantech")) %>% as_vector()

df_reg <- df_model %>% 
  mutate(tech_prox = tech_prox*100) 
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- polr(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, method = "logistic")
  df_temp <- clean_results(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res)
df_latex

print.xtable(df_latex, 
             type = "latex", 
             #include.rownames = FALSE,
             #floating = FALSE, 
             format.args = list(digits = 3, big.mark = " ", decimal.mark = ","))






## Logit ==================================================================
### Umweltwirkung #########################################################
### TechProx

model_equations <- c(  
  "umwelt_wirk_bin ~ tech_prox + branche11",
  #"umwelt_wirk_bin ~ tech_prox + branche11 + log(bes)",
  "umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age",
  #"umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue",
  #"umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes",
  "umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe",
  #"umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj",
  "umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj",
  #"umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam",
  "umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni",
  "umwelt_wirk_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni + produkt_kat"
)

df_reg <- df_model %>% 
  mutate(tech_prox = tech_prox*100) 
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- glm(formula = as.formula(model_equation), data = df_reg, na.action = na.omit, family = binomial(link = "logit"))
  df_temp <- clean_results_logit(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res)
df_latex

print.xtable(df_latex, 
             type = "latex", 
             #include.rownames = FALSE,
             #floating = FALSE, 
             format.args = list(digits = 3, big.mark = " ", decimal.mark = ","))


### cleantech

model_equations <- lapply(model_equations, function(x) str_replace(x, "tech_prox", "cleantech")) %>% as_vector()

df_reg <- df_model
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- glm(formula = as.formula(model_equation), data = df_reg, na.action = na.omit,  family = binomial(link = "probit"))
  df_temp <- clean_results_logit(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_bottom <- df_res %>% filter(variable %in% bottom)
df_top <- df_res %>% filter(!(variable %in% bottom))

df_res <- df_top %>% 
  bind_rows(df_bottom)



df_latex <- df_res %>% 
  filter(!str_detect(variable, "branche.{1,}")) %>% 
  filter(!str_detect(variable, "produkt.{1,}")) %>% 
  filter(!str_detect(variable, "ja"))

print.xtable(df_latex, type = "latex")


### Umweltinnovation ######################################################
### TechProx

model_equations <- c(  
  "umwelt_inno_bin ~ tech_prox + branche11",
  #"umwelt_inno_bin ~ tech_prox + branche11 + log(bes)",
  "umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age",
  #"umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue",
  #"umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes",
  "umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe",
  #"umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj",
  "umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj",
  #"umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam",
  "umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni",
  "umwelt_inno_bin ~ tech_prox + branche11 + log(bes) + age + fue + fuep_bes + foe + umsj + gewj + anzteam + uni + produkt_kat"
)

df_reg <- df_model #%>% 
  #mutate(tech_prox = tech_prox*100) 
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- glm(formula = as.formula(model_equation), data = df_reg, na.action = na.omit,  family = binomial(link = "logit"))
  df_temp <- clean_results_logit(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable")
  }
}

df_latex <- create_reg_tab(df_res)
df_latex <- df_latex %>% mutate(variable = factor(variable, c("\\textsc{CleanTech}", "log(size)", "age", "subsidy", "R\\&D", "R\\&D intensity", 
                                    "returns", "break even", "team size", "university", "Sector controls", "Product type controls", "$N$", "Pseudo $R^2$"))) %>% 
  arrange(variable)

print.xtable(df_latex, 
             type = "latex", 
             #include.rownames = FALSE,
             #floating = FALSE, 
             format.args = list(digits = 3, big.mark = " ", decimal.mark = ","))


### cleantech

model_equations <- lapply(model_equations, function(x) str_replace(x, "tech_prox", "cleantech")) %>% as_vector()

df_reg <- df_model
df_res <- tibble()
bottom <- c("N", "BIC", "Pseudo_R2")
for (i in seq_along(model_equations)){
  model_equation <- model_equations[i]
  polr_mod <- glm(formula = as.formula(model_equation), data = df_reg, na.action = na.omit,  family = binomial(link = "logit"))
  df_temp <- clean_results_logit(model = polr_mod)
  colnames(df_temp)  <- c("variable", i)
  if(i == 1){
    df_res <- df_temp
  }
  else{
    df_res <- df_res %>% full_join(df_temp, by = "variable", )
  }
}

df_latex <- create_reg_tab(df_res)
df_latex <- df_latex %>% mutate(variable = factor(variable, c("\\textsc{TechProx}$_{max}$", "log(size)", "age", "subsidy", "R\\&D", "R\\&D intensity", 
                                                              "returns", "break even", "team size", "university", "Sector controls", "Product type controls", "$N$", "Pseudo $R^2$"))) %>% 
  arrange(variable)

print.xtable(df_latex, 
             type = "latex", 
             #include.rownames = FALSE,
             #floating = FALSE, 
             format.args = list(digits = 3, big.mark = " ", decimal.mark = ","))





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

