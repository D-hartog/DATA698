library(tidymodels)
library(timetk)
library(fpp3)

library(modeltime)
library(modeltime.ensemble)
library(tidyverse)
library(skimr)

library(xgboost)
library(lightgbm)

# LOAD DATA

daily_counts_type <- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/brooklyn_dailyCIP_counts.csv")
holidays <- read_csv("https://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Holidays_cleaned.csv")
weather_data <- read_csv("hhttps://raw.githubusercontent.com/D-hartog/DATA698/refs/heads/main/Project%20files/Data/Weather_data_cleaned.csv")

head(holidays)
holiday_dates <- unique(holidays$Date)

## Re-code cip_jobs from 4 categories to 2

daily_counts_type <- daily_counts_type %>% 
  mutate(call_type = case_when(
    cip_jobs == "Non CIP" ~ "Non CIP",
    cip_jobs == "Non Critical" ~ "CIP",
    cip_jobs == "Serious" ~ "CIP",
    cip_jobs == "Critical" ~ "CIP"))

## Convert precinct and call_type to factor
daily_counts_type$call_type <- factor(daily_counts_type$call_type)
daily_counts_type$nypd_pct_cd <- factor(daily_counts_type$nypd_pct_cd)

## Groupby precinct/date/cip_type and calculate call counts
daily_counts_type_grouped <- daily_counts_type %>% 
  group_by(nypd_pct_cd, incident_date, call_type) %>% 
  summarise(count = sum(counts)) %>% ungroup()

## Fill in missing dates 

daily_counts_type_ts <- as_tsibble(daily_counts_type_grouped,  index = incident_date, key = c(nypd_pct_cd, call_type))
daily_counts_type_ts <- daily_counts_type_ts %>% fill_gaps(.full = TRUE, count = 0)
daily_counts_type_ts <- as_tibble(daily_counts_type_ts)

skim(daily_counts_type_ts)


## Plot a sample of precincts 

precint_sample <- sample(unique(daily_counts_type_ts$nypd_pct_cd), 4)

## MODELTIME PLOT

# daily_counts_type_ts %>% filter(nypd_pct_cd %in% precint_sample) %>% 
#   plot_time_series(incident_date, count,
#                    .smooth= FALSE,
#                    .smooth_period = 14,
#                    .facet_vars = c(nypd_pct_cd, call_type),
#                    .facet_ncol = 2, 
#                    .interactive = FALSE, 
#                    .line_color = "black") +
#   theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
#         panel.background = element_rect(fill = "white"),
#         strip.background = element_rect(fill = "lightgrey"),
#         panel.grid.major = element_line(color = "grey", linewidth = 0.5),
#         panel.grid.minor = element_blank(), # Removes minor grid lines
#         strip.text = element_text(colour = "black")) + 
#   labs(title = "Sample Brooklyn Precinct Time Series")

daily_counts_type_ts %>% filter(nypd_pct_cd %in% precint_sample) %>% 
  ggplot(aes(x = incident_date, y = count)) + 
  geom_line() +
  facet_wrap(vars(nypd_pct_cd, call_type), nrow = 4, ncol = 2, scales = "free_y") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) + 
  labs(title = "Brooklyn: Sample Precinct Time Series",
       x = "Date", y = "Number of calls")


# Zoomed in Data from sample Precincts - October 2024
daily_counts_type_ts %>% filter(nypd_pct_cd %in% c(62,69,75,83), 
                                (incident_date > "2024-09-30" & 
                                  incident_date < "2024-11-03")) %>% 
  ggplot(aes(x = incident_date, y = count)) + 
  geom_line() +
  facet_wrap(vars(nypd_pct_cd, call_type), nrow = 4, ncol = 2, scales = "free_y") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) + 
  labs(title = "Brooklyn: Sample Precinct Time Series",
       x = "Date", y = "Number of calls")


# Prepare data

daily_counts_type_ts$boro_id <- "Brooklyn"
# daily_counts_type_ts$type_precinct_id <- paste0(daily_counts_type_ts$call_type, 
#                                                 "_", 
#                                                 daily_counts_type_ts$nypd_pct_cd)

daily_counts_type_ts <- daily_counts_type_ts %>% pivot_longer(
  cols = c(nypd_pct_cd, boro_id), 
  names_to = "category", 
  values_to = "identifier") %>% 
  group_by(call_type, category, identifier, incident_date) %>% 
  summarise(count = sum(count)) %>% ungroup()


# FEATURE ENGINEERING  

#daily_counts_type_ts <- daily_counts_type_ts %>% group_by(nypd_pct_cd, call_type) %>%

daily_counts_type_ts <- daily_counts_type_ts %>% group_by(category, identifier) %>%
  # Extend into the future
  future_frame(incident_date,
               .length_out = 7,
               .bind_data = TRUE) %>%
  # Add Time series features
  tk_augment_lags(count, .lags = c(1,7,14)) %>%
  tk_augment_slidify(
    count_lag1,
    count_lag7,
    count_lag14,
    .f = ~ mean(., na.rm = TRUE),
    .period = c(7,14,28,28*2),
    .align = "right",
    .partial = TRUE
  ) %>%
  ungroup() %>% rowid_to_column(var = "row_id")

glimpse(daily_counts_type_ts)
skim(daily_counts_type_ts)


# PREPARE DATA
## Remove missing values
training_data <- daily_counts_type_ts %>% 
  filter(!is.na(count)) %>% 
  filter(!is.na(count_lag14))

## Add holiday variable 
training_data <- training_data %>% 
  mutate(holiday = as_factor(ifelse(incident_date %in% holiday_dates, 1, 0)))

## Add weather variables
# Mutate new variables
weather_data <- weather_data %>% mutate(Prec = as.factor(ifelse(Precipitation <= 0.01, 0, 1)),
                                        SF = as.factor(ifelse(Snowfall <= 0.1 , 0, 1)),
                                        SD = as.factor(ifelse(SnowDepth <= 1, 0, 1)))

weather_data_vars <- weather_data %>% select(Date, AvgTemperature, Prec, SF)

training_data <- training_data %>% 
  left_join(weather_data_vars, by = join_by(incident_date == Date))

training_data <- training_data %>% mutate(dow = factor(weekdays(incident_date)))

## Create future data to forecast
future_data <- daily_counts_type_ts %>% filter(is.na(count))


# TRAIN/TEST SPLIT
# Test data is the last week
splits <- training_data %>% time_series_split(incident_date, 
                                                     assess = 7, 
                                                     cumulative = TRUE)

splits %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(incident_date, count,
                           .interactive = FALSE, 
                           .line_color = "black")  +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) + 
  labs(title = "Time Series Train/Test SPlit",
       x = "Date", y = "Number of calls")

# Recipe
## Original 
# xgb_recipe <- recipe(formula = count ~., data = training(splits)) %>% 
#   update_role(row_id, incident_date, new_role = "id") %>% 
#   step_timeseries_signature(incident_date) %>%
#   step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
#   step_dummy(all_nominal_predictors(), one_hot = TRUE)

## Modified
xgb_recipe_mod <- recipe(formula = count ~., data = training(splits)) %>% 
  update_role(row_id, incident_date, category, identifier, call_type, new_role = "id") %>% 
  # step_timeseries_signature(incident_date) %>%
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)|(.num)")) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

### SWITCH OUT THE DIFFERENT RECIPES ###

xgb_recipe_mod %>% summary()
# Adds in engineer variables based on data
xgb_recipe_mod %>% prep() %>% summary()

# Save the recipe
recipe_summary <- xgb_recipe_mod %>% prep() %>% summary() 
recipe_summary %>% print(n = 46)
write_csv(recipe_summary, "xgb_recipe_summary_bk.csv")

xgb_recipe_mod %>% prep() %>% juice() %>% glimpse()

# Machine Learning - specify model
xgboost <- boost_tree(
  trees = 1000,
  learn_rate = 0.05
)

# Default loss function that penalizes underestimates
xgboost_default_wkfl <- workflow() %>% 
  add_model(xgboost %>% 
              set_engine("xgboost", objective = "reg:squarederror") %>%
              set_mode("regression")) %>% 
  add_recipe(xgb_recipe_mod) %>%
  fit(training(splits))

# Add in a different loss function that penalizes underestimates
xgboost_tweedie_wkfl <- workflow() %>% 
  add_model(xgboost %>% 
              set_engine("xgboost", objective = "reg:tweedie") %>%
              set_mode("regression")) %>% 
  add_recipe(xgb_recipe_mod) %>%
  fit(training(splits))

# Add in Poisson regression
xgboost_poisson_wkfl <- workflow() %>% 
  add_model(xgboost %>% 
              set_engine("xgboost", 
                         objective = "count:poisson",
                         eval_metric = "poisson-nloglik") %>%
              set_mode("regression")) %>% 
  add_recipe(xgb_recipe_mod) %>%
  fit(training(splits))


# Variable Importance
## Previous library used to get VIP: library(caret)

library(vip)
xgboost_tweedie_wkfl %>% 
  extract_fit_parsnip() %>%
  vip(num_features = 10, aesthetics = list(fill = "black")) + 
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) + 
  labs(title = "Top 10 Important Features: Tweedie")

xgboost_default_wkfl %>% 
  extract_fit_parsnip() %>%
  vip(num_features = 10, aesthetics = list(fill = "black")) +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(title = "Top 10 Important Features: SqrError")

xgboost_poisson_wkfl %>% 
  extract_fit_parsnip() %>%
  vip(num_features = 10, aesthetics = list(fill = "black")) + 
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) + 
  labs(title = "Top 10 Important Features: Poisson")

# Modeltime
models_tbl <- modeltime_table(
  xgboost_default_wkfl, 
  xgboost_tweedie_wkfl, 
  xgboost_poisson_wkfl
) %>% 
  mutate(.model_desc = case_when(.model_id == 1 ~ str_c(.model_desc, " - SqEr"),
                                 .model_id == 2 ~ str_c(.model_desc, " - Tweedie"),
                                 .model_id == 3 ~ str_c(.model_desc, " - Poisson")))

# Calibrate on test
calibration_tbl <- models_tbl %>% 
  modeltime_calibrate(new_data = testing(splits)) 

## Calculate accuracy metrics 
calibration_tbl %>%
  modeltime_accuracy() 

# TEST FORECAST TABLE
test_forecast_table <- calibration_tbl %>% modeltime_forecast(
  new_data = testing(splits), 
  actual_data = training_data, 
  keep_data = TRUE)

  
write_csv(test_forecast_table, "XGBOOST_forecast_table_bk.csv")
test_forecast_table %>% plot_modeltime_forecast()

# test_forecast_table <- models_tbl %>%
#   modeltime_forecast(
#     new_date = testing(splits),
#     actual_data = training_data,
#     keep_data = TRUE
#   )

# Visualize and filter
unique(training_data$category)

## Boro
filter_identifier_boro <- "Brooklyn"

test_forecast_table %>% filter(identifier == filter_identifier_boro) %>% 
  filter_by_time(
    .date_var = .index,
    .start_date = "2025-06-15",
    .end_date = "2025-06-30") %>% 
  plot_modeltime_forecast(
    .facet_vars = call_type,
    #.facet_ncol = 2,
    .facet_nrow = 2, 
    .conf_interval_show = TRUE,
    .conf_interval_alpha = 0.15,
    #.line_color = "black"
    .interactive = FALSE) +
   scale_discrete_manual(values = c("black","blue","yellow","red"),
                         aesthetics = "colour") +
  labs(title = "XGBoost - Brooklyn") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey",
                                        color = "lightgrey",
                                        linewidth = 0),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black"))

unique(test_forecast_table$.model_desc)
## Precinct level
filter_precinct <- training_data %>% filter(category == "nypd_pct_cd") %>% 
  distinct(identifier) %>% pull()

## Call type level

filter_category_type <- "nypd_pct_cd"


test_forecast_table %>% filter(identifier %in% c("62", "69", "75", "83")) %>% 
  group_by(identifier, call_type) %>% 
  filter_by_time(
    .date_var = .index,
    .start_date = "2025-06-13",
    .end_date = "end") %>% 
  plot_modeltime_forecast(
    .facet_vars = call_type,
    .facet_ncol = 2,
    .facet_nrow = 4, 
    .conf_interval_show = TRUE,
    .conf_interval_alpha = 0.15,
    .interactive = FALSE) +
    scale_discrete_manual(values = c("black","blue","yellow","red"),
                          aesthetics = "colour") + 
    labs(title = "XGBoost - Precinct") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey",
                                        color = "lightgrey",
                                        linewidth = 0),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black"))
  

test_forecast_table %>% filter(identifier == 62) %>% 
  group_by(identifier, call_type) %>% 
  filter_by_time(
    .date_var = .index,
    .start_date = "2025-06-13",
    .end_date = "end") %>% 
  plot_modeltime_forecast(
    .facet_vars = call_type,
    .facet_ncol = 3,
    .facet_nrow = 2, 
    .conf_interval_show = TRUE,
    .conf_interval_alpha = 0.15,
    .interactive = FALSE) +
  scale_discrete_manual(values = c("black","blue","yellow","red"),
                        aesthetics = "colour") + 
  labs(title = "XGBoost - Precinct 62") +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey",
                                        color = "lightgrey",
                                        linewidth = 0),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black"))



## Accuracy by identifier

accuracy_by_identifier <- test_forecast_table %>% filter %>% 
  select(.model_desc, category, identifier, call_type, .index, .value) %>% 
  pivot_wider(
    names_from = `.model_desc`,
    values_from = .value) %>% 
  filter(!is.na(`XGBOOST - SqEr`)) %>% 
  pivot_longer(cols = `XGBOOST - SqEr` : `XGBOOST - Poisson`) %>% 
  group_by(identifier, call_type, name) %>% 
  summarize_accuracy_metrics(
    truth = ACTUAL, 
    estimate = value,
    metric_set = default_forecast_accuracy_metric_set()
  )

accuracy_by_identifier %>% print(n = 93)
write_csv(accuracy_by_identifier, "BK_XGBOOST_accuracy.csv")

best_rmse <- accuracy_by_identifier %>% group_by(identifier, call_type) %>% 
  slice_min(rmse, n = 1) %>% ungroup()

write_csv(best_rmse, "BK_XGBOOST_best_rmse.csv")

## Model Summary

# Extract full workflow model: Poisson 
poisson_wf_fit <- xgboost_poisson_wkfl %>% extract_fit_parsnip()

# Extract the xgboost booster
poisson_booster <- poisson_wf_fit$fit

##  Model dump
## xgb.dump(poisson_booster, with_stats = TRUE)

# Params
poisson_parameters <- tibble(Parmeters = names(poisson_booster$params), 
                             Value = unlist(poisson_booster$params))


# Extract full workflow model: Default - SqEr 
sqer_wf_fit <- xgboost_default_wkfl %>% extract_fit_parsnip()

# Extract the xgboost booster
sqer_booster <- sqer_wf_fit$fit

# Model dump
## xgb.dump(poisson_booster, with_stats = TRUE)

# Params
sqer_parameters <- tibble(Parmeters = names(sqer_booster$params), 
                             Value = unlist(sqer_booster$params))

# Extract full workflow model: Tweedie 
tweedie_wf_fit <- xgboost_tweedie_wkfl %>% extract_fit_parsnip()

# Extract the xgboost booster
tweedie_booster <- tweedie_wf_fit$fit

# Model dump
## xgb.dump(poisson_booster, with_stats = TRUE)

# Params
tweedie_parameters <- tibble(Parmeters = names(tweedie_booster$params), 
                          Value = unlist(tweedie_booster$params))

write_csv(poisson_parameters, "brooklyn_fit_params.csv")

# Results visualization
best_rmse %>% ggplot(aes(y = fct_rev(fct_infreq(name)))) + geom_bar(fill = "black") + 
  facet_wrap(vars(call_type)) +
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        panel.background = element_rect(fill = "white"),
        strip.background = element_rect(fill = "lightgrey"),
        panel.grid.major = element_line(color = "grey", linewidth = 0.5),
        panel.grid.minor = element_blank(), # Removes minor grid lines
        strip.text = element_text(colour = "black")) +
  labs(title = "Model with Lowest RMSE: Brooklyn", x = "Model", y = "Number of models")





