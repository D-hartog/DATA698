library(tidymodels)
library(timetk)
library(fpp3)

library(modeltime)
library(modeltime.ensemble)
library(tidyverse)
library(skimr)

library(xgboost)
library(lightgbm)

# Load in data 

daily_counts_type <- read_csv("/Users/dirkhartog/Desktop/CUNY_MSDS/DATA_698/Data/manhattan/manhattan_dailyCIP_counts.csv")
holidays <- read_csv("/Users/dirkhartog/Desktop/CUNY_MSDS/DATA_698/Holidays_cleaned.csv")

head(holidays)
holiday_dates <- unique(holidays$Date)

# Re-code cip_jobs from 4 categories to 2

daily_counts_type <- daily_counts_type %>% 
  mutate(call_type = case_when(
    cip_jobs == "Non CIP" ~ "Non CIP",
    cip_jobs == "Non Critical" ~ "CIP",
    cip_jobs == "Serious" ~ "CIP",
    cip_jobs == "Critical" ~ "CIP"))

daily_counts_type$call_type <- factor(daily_counts_type$call_type)
daily_counts_type$nypd_pct_cd <- factor(daily_counts_type$nypd_pct_cd)

# Groupby precinct/date/cip_type and calculate call counts

daily_counts_type_grouped <- daily_counts_type %>% 
  group_by(nypd_pct_cd, incident_date, call_type) %>% 
  summarise(count = sum(counts)) %>% ungroup()

# Fill in missing dates 

daily_counts_type_ts <- as_tsibble(daily_counts_type_grouped,  index = incident_date, key = c(nypd_pct_cd, call_type))
daily_counts_type_ts <- daily_counts_type_ts %>% fill_gaps(.full = TRUE, count = 0)
daily_counts_type_ts <- as_tibble(daily_counts_type_ts)

skim(daily_counts_type_ts)

# Feature Engineering

# daily_counts_type_ts <- daily_counts_type_ts %>% mutate(month = as_factor(month(incident_date)),
#                                       dow = as_factor(weekdays(incident_date)),
#                                       weekend = as_factor(ifelse(wday(incident_date) %in% c(1,7), 1, 0)),
#                                       holiday = as_factor(ifelse(incident_date %in% holiday_dates, 1, 0))
#)


# Plot a sample of precincts 

precint_sample <- sample(unique(daily_counts_type_ts$nypd_pct_cd), 4)

daily_counts_type_ts %>% filter(nypd_pct_cd %in% precint_sample) %>% 
  plot_time_series(incident_date, count,
                   .smooth= TRUE,
                   .smooth_period = 14,
                   .facet_vars = c(nypd_pct_cd, call_type),
                   .facet_ncol = 2) 

  
# Split data into training and testing sets

# training_ts <- daily_counts_type_ts %>% 
#   filter(incident_date <= "2025-06-15")
# 
# testing_ts <- daily_counts_type_ts %>% 
#   filter(incident_date > "2025-06-15")

# Feature Engineering 

training_ts <- daily_counts_type_ts %>% group_by(nypd_pct_cd, call_type) %>% 
  future_frame(incident_date, 
               .length_out = 15, 
               .bind_data = TRUE) %>% 
  tk_augment_lags(count, .lags = 28) %>% 
  tk_augment_slidify(
    count_lag28,
    .f = ~ mean(., na.rm = TRUE),
    .period = c(7,14,28,28*2),
    .align = "center",
    .partial = TRUE
  ) %>%
  ungroup() %>% rowid_to_column(var = "row_id")

glimpse(training_ts)

# Prepare Data
## Remove missing values

training_ts_prepared <- training_ts %>% filter(!is.na(count_lag28))

future_data <- training_ts %>% filter(is.na(count))
# future_data <- future_data %>% mutate(month = as_factor(month(incident_date)),
#                                                         dow = as_factor(weekdays(incident_date)),
#                                                         weekend = as_factor(ifelse(wday(incident_date) %in% c(1,7), 1, 0)),
#                                                         holiday = as_factor(ifelse(incident_date %in% holiday_dates, 1, 0))
# )

# Time Split

splits <- training_ts_prepared %>% time_series_split(incident_date, 
                                                     assess = 30, 
                                                     cumulative = TRUE)

splits %>% tk_time_series_cv_plan() %>% plot_time_series_cv_plan(incident_date, count)

# Recipe

xgb_recipe <- recipe(count ~., data = training(splits)) %>% 
  update_role(row_id, incident_date, nypd_pct_cd, call_type, new_role = "id") %>% 
  step_timeseries_signature(incident_date) %>%
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

xgb_recipe %>% summary()
xgb_recipe %>% prep() %>% summary() %>% 
recipe_summary <- xgb_recipe %>% prep() %>% summary() 
recipe_summary$type
write_csv(recipe_summary, "xgb_recipe_summary.csv")

xgb_recipe %>% prep() %>% juice() %>% glimpse()

# Machine Learning - specifiy model
xgboost <- boost_tree(
  trees = 1000,
  learn_rate = 0.05
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")


# xgboost <- workflow() %>% add_model(
#   boost_tree(mode = "regression") %>% # We can specify the tress i
#     set_engine("xgboost")) %>%
#   add_recipe(xgb_spec) %>% 
#   fit(training(splits))

# Bundle model + recipe
wf_xgb <- workflow() %>% 
  add_model(xgboost) %>% 
  add_recipe(xgb_recipe)

# Fit Model

xgb_fit <- wf_xgb %>% fit(training(splits))

# Variable Importance
## Previous library used to get VIP: library(caret)

library(vip)
xgb_fit %>% 
  extract_fit_parsnip() %>%
  vip(num_features = 10) + 
  theme_minimal() + 
  labs(title = "Top 10: Variable Importance Factor")

# Modeltime

## Calibrate on test

model_tbl <- modeltime_table(
  xgb_fit)

model_tbl %>% modeltime_calibrate(testing(splits)) %>% modeltime_accuracy()

# Forecast

forecast_table <- model_tbl %>% 
  modeltime_forecast(
    new_date = testing(splits),
    actual_data = training_ts_prepared,
    keep_data = TRUE
  )

# Visualize 

## Manhattan Total
boro_agg <- forecast_table %>% group_by(incident_date, .model_desc, call_type) %>% 
  summarise(counts = sum(.value))

# forecast_table %>% filter(.model_desc == "ACTUAL") %>% 
#   group_by(incident_date, .model_desc, call_type) %>% summarise(counts = sum(.value)) %>% 
#   ggplot(aes(x = incident_date, y = counts, color = .model_desc)) + geom_line() + 
#   geom_line(data = forecast_table %>% filter(.model_desc == "XGBOOST") %>%
#               group_by(incident_date, .model_desc, call_type) %>% summarise(counts = sum(.value)), 
#             aes(x = incident_date, y = counts), alpha = 0.5) + 
#   facet_grid(rows = vars(call_type), scales = "free_y")

boro_agg %>% filter(.model_desc == "ACTUAL") %>% 
  ggplot(aes(x = incident_date, y = counts, color = .model_desc)) + geom_line() + 
  geom_line(data = boro_agg %>% filter(.model_desc == "XGBOOST"), 
            aes(x = incident_date, y = counts), alpha = 0.5) + 
  facet_grid(rows = vars(call_type), scales = "free_y") + 
  labs(title = "Fitted Model and Forecasts: Manhattan", 
       x = "Date", y = "Number of calls") +
  theme_minimal()

# Precinct total
precinct_agg <-  forecast_table %>% group_by(incident_date, nypd_pct_cd, .model_desc, call_type) %>% 
  summarise(counts = sum(.value)) %>% filter(incident_date > "2025-04-01")

head(precinct_agg, 10)

## Visualize each precinct separately 
precinct_agg %>% filter(.model_desc == "ACTUAL", nypd_pct_cd %in% c(5,6,7)) %>% 
  ggplot(aes(x = incident_date, y = counts, color = .model_desc)) + geom_line() + 
  geom_line(data = precinct_agg %>% 
              filter(.model_desc == "XGBOOST", nypd_pct_cd %in% c(5,6,7)), 
            aes(x = incident_date, y = counts), alpha = 0.5) + 
  facet_wrap(nypd_pct_cd ~ call_type, nrow = 3, ncol = 2, scales = "free_y")  + 
  labs(title = "Fitted Model and 15 day Forecasts: Precinct 5, 6, 7", 
       x = "Date", y = "Number of calls") +
  theme_minimal()
