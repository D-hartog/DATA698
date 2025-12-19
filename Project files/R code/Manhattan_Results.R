library(tidyverse)

# _________ARIMA/ETS_________

model_report <- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Results%20data/manhattan_model_report.csv")
manhattan_fc_accuracy <- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Results%20data/manhattan_fc_accuracy.csv")

head(manhattan_fc_accuracy)


best_rmse <- manhattan_fc_accuracy %>% group_by(nypd_pct_cd, call_type) %>% slice_min(RMSE, n = 1)

best_rmse %>% ggplot(aes(y = fct_rev(fct_infreq(.model)))) + geom_bar(fill = "black") + 
  facet_wrap(vars(call_type)) +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(title = "Model with Lowest RMSE Among ARIMA/ETS Candidates\nManhattan Data Set", 
       x = "Number of Models", y = "Model Description")


# Get model report info 

Best_AICc <- model_report %>% select(nypd_pct_cd, call_type, .model, AIC, AICc, BIC) %>% 
  group_by(nypd_pct_cd, call_type) %>% slice_min(AICc, n = 1)

Best_AICc %>% filter(.model %in% c("arima_weather", "base_arima")) %>% 
  ggplot(aes(y = fct_rev(fct_infreq(.model)))) + geom_bar(fill = "black") + 
  facet_wrap(vars(call_type)) +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(title = "Models with Lowest AICc Among ARIMA/ETS Candidates\nManhattan Data Set", 
       x = "Number of Models", y = "Model Description")


## Get diagnositcs tests for each model ##
lb_pval <- model_report %>% select(nypd_pct_cd, call_type, .model, lb_pvalue)
lb_pval <- lb_pval %>% arrange(.model, call_type, nypd_pct_cd)

acf1 <- manhattan_fc_accuracy %>% select(nypd_pct_cd, call_type, .model, ACF1)
acf1 <- acf1 %>% arrange(.model, call_type, nypd_pct_cd)

# Combine the two diagnostic test results 
combined_diagnostics <- lb_pval %>% left_join(acf1, by = c("nypd_pct_cd", "call_type", ".model"))

# Create new column white_noise if lb_pvalue > 0.05
combined_diagnostics <- combined_diagnostics %>% mutate(white_noise = ifelse(lb_pvalue > 0.05, "TRUE", "FALSE"))

# combined_diagnostics %>% select(nypd_pct_cd, call_type, .model, white_noise) %>% 
#   filter(call_type == "CIP") %>% select(-call_type) %>% 
#   pivot_wider(names_from = ".model", values_from = "white_noise")

combined_diagnostics %>% select(nypd_pct_cd, call_type, .model, white_noise) %>% 
  filter(call_type == "CIP") %>% select(-call_type) %>% 
  ggplot(aes(x = .model, y = nypd_pct_cd, fill = white_noise)) + 
  geom_tile(color = "black") +
  scale_fill_manual(values = c("TRUE" = "#D55E00","FALSE" ="#56B4E9")) + 
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(x = "Model name", y = "Precinct", title = "CIP Models with White Noise Residuals")

combined_diagnostics %>% select(nypd_pct_cd, call_type, .model, white_noise) %>% 
  filter(call_type == "Non CIP") %>% select(-call_type) %>% 
  ggplot(aes(x = .model, y = nypd_pct_cd, fill = white_noise)) + 
  geom_tile(color = "black") +
  scale_fill_manual(values = c("TRUE" = "#D55E00","FALSE" ="#56B4E9")) + 
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(x = "Model name", y = "Precinct", title = "Non CIP Models with White Noise Residuals")

# _________XGBoost_________

xg_recipe<- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Results%20data/xgb_recipe_summary_man.csv")
manhattan_fc_accuracy_xg <- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Results%20data/MAN_XGBOOST_accuracy.csv")
xg_best_rmse <- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Results%20data/MAN_XGBOOST_best_rmse.csv")

# Select and rename columns 

xg_best_rmse <- xg_best_rmse %>% select(identifier, call_type, name, rmse) %>% 
  rename("nypd_pct_cd" = "identifier", "RMSE" = "rmse")

best_rmse <- best_rmse %>% select(nypd_pct_cd, call_type, .model, RMSE) %>%
  rename("name" = ".model")


combined_rmse <- rbind(best_rmse, xg_best_rmse)

combined_rmse <- combined_rmse %>% arrange(nypd_pct_cd, call_type, name)
combined_rmse %>% group_by(nypd_pct_cd, call_type) %>% slice_min(RMSE, n = 1) %>% print(n=70)

comparisons <- combined_rmse %>% mutate(model_name = ifelse(name %in% c("XGBOOST - SqEr",
                                                         "XGBOOST - Poisson",
                                                         "XGBOOST - Tweedie"), "XG", "ARIMA/ETS"))

comparisons %>% filter(call_type != "<aggregated>") %>% 
  ggplot(aes(y = fct_rev(fct_infreq(model_name)))) + geom_bar(fill = "black") + 
  facet_wrap(vars(call_type)) +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(title = "Model with Lowest RMSE", x = "Model", y = "Number of models")

