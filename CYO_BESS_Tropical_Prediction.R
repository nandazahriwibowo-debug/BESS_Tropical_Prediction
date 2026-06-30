# =============================================================================
# CYO Project: Predictive Thermal Runaway and Capacity Degradation Strategy
# for Tropical Industrial Lithium-Ion Battery Storage Systems
#
# Author: Nanda Zahri Wibowo
# Institution: Indonesia
# Course: Harvard X - Choose Your Own (CYO) Capstone
#
# This script builds a realistic dataset for industrial-scale lithium-ion
# battery energy storage systems (BESS) operating under tropical climatic stress,
# then implements a multi-model predictive pipeline for both thermal runaway
# classification and capacity degradation regression.
# =============================================================================

# ---- 1. LIBRARY INITIALIZATION ------------------------------------------------

# Suppress noisy startup messages from packages that love to talk
suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(randomForest)
  library(xgboost)
  library(corrplot)
  library(gridExtra)
  library(pROC)
})

# Fixed seed for reproducibility. I picked 202406 because that's when I started
# drafting this project.
set.seed(202406)

# ---- 2. SIMULATED DATA GENERATION ---------------------------------------------

# No access to proprietary BESS operational logs from Southeast Asian utilities.
# So I built a simulation grounded in physics, informed by literature on NMC811
# pouch cells and LFP cylindrical cells. Parameters come from published
# accelerated aging studies (Feng et al., 2022; Ketjoy et al., 2025).

generate_bess_data <- function(n_cells = 2800) {

  # Cell identifiers and chemistry labels
  cell_id <- sprintf("CELL_%04d", seq_len(n_cells))
  chemistry <- sample(c("NMC811", "LFP"), n_cells, replace = TRUE, prob = c(0.55, 0.45))

  # Tropical ambient conditions. Maritime Southeast Asia averages 27-32 C,
  # with daily peaks past 38 C in poorly ventilated industrial sheds.
  ambient_temp <- rgamma(n_cells, shape = 12, scale = 2.6) + 18
  ambient_temp <- pmin(pmax(ambient_temp, 22), 48)

  relative_humidity <- rbeta(n_cells, shape1 = 3.2, shape2 = 1.8) * 60 + 35
  relative_humidity <- pmin(pmax(relative_humidity, 40), 98)

  # Charge throughput. Industrial BESS typically cycles 0.8-1.5 times per day.
  cycle_count <- rpois(n_cells, lambda = 1850) + rnorm(n_cells, 200, 80)
  cycle_count <- pmax(cycle_count, 50)

  avg_c_rate <- rnorm(n_cells, mean = 0.85, sd = 0.22)
  avg_c_rate <- pmin(pmax(avg_c_rate, 0.2), 2.5)

  # Voltage stress metrics — chemistry-specific
  max_charge_voltage <- ifelse(chemistry == "NMC811",
                                 rnorm(n_cells, 4.18, 0.04),
                                 rnorm(n_cells, 3.55, 0.03))
  min_discharge_voltage <- ifelse(chemistry == "NMC811",
                                  rnorm(n_cells, 2.65, 0.08),
                                  rnorm(n_cells, 2.35, 0.06))

  # Internal resistance grows non-linearly with aging and temperature exposure
  base_ir <- ifelse(chemistry == "NMC811", 0.018, 0.012)
  internal_resistance <- base_ir * (1 + 0.00045 * cycle_count + 0.003 * pmax(ambient_temp - 30, 0))
  internal_resistance <- internal_resistance * rlnorm(n_cells, meanlog = 0, sdlog = 0.15)

  # Surface temperature is a function of ambient, internal resistance, and C-rate
  surface_temp <- ambient_temp + 2.5 * avg_c_rate + 18 * internal_resistance * avg_c_rate + rnorm(n_cells, 0, 1.2)

  # Differential temperature: gap between surface and ambient
  delta_temp <- surface_temp - ambient_temp

  # Electrolyte degradation proxy: latent composite influenced by humidity,
  # high-voltage exposure, and temperature excursions
  electrolyte_degradation <- 0.15 * relative_humidity / 100 +
    0.35 * (max_charge_voltage - ifelse(chemistry == "NMC811", 4.2, 3.6)) / 0.6 +
    0.25 * pmax(ambient_temp - 35, 0) / 15 +
    0.25 * cycle_count / 5000 +
    rnorm(n_cells, 0, 0.04)

  # State of Health (SOH) in percent. Capacity fades faster in tropical heat.
  # Chemistry-specific base decay rates calibrated to literature.
  soh_base <- 100 - ifelse(chemistry == "NMC811",
                           0.0038 * cycle_count + 0.12 * pmax(ambient_temp - 25, 0),
                           0.0022 * cycle_count + 0.07 * pmax(ambient_temp - 25, 0))

  soh <- soh_base - 8 * electrolyte_degradation + rnorm(n_cells, 0, 1.5)
  soh <- pmin(pmax(soh, 58), 100)

  # Capacity fade percentage (complementary to SOH)
  capacity_fade <- 100 - soh

  # Thermal runaway (TR) is a rare binary event. Modeled via a latent
  # probability combining surface temperature, electrolyte degradation, voltage
  # abuse, and internal resistance. The logit ensures physical rarity.
  tr_logit <- -6.8 +
    0.18 * (surface_temp - 35) +
    2.4 * electrolyte_degradation +
    3.2 * (max_charge_voltage - ifelse(chemistry == "NMC811", 4.15, 3.5)) +
    12 * (internal_resistance - 0.015) +
    0.0015 * cycle_count +
    rnorm(n_cells, 0, 0.6)

  tr_prob <- 1 / (1 + exp(-tr_logit))
  thermal_runaway <- ifelse(runif(n_cells) < tr_prob, 1, 0)

  # Pack together
  tibble(
    cell_id = cell_id,
    chemistry = chemistry,
    ambient_temp = round(ambient_temp, 2),
    relative_humidity = round(relative_humidity, 1),
    cycle_count = round(cycle_count, 0),
    avg_c_rate = round(avg_c_rate, 3),
    max_charge_voltage = round(max_charge_voltage, 3),
    min_discharge_voltage = round(min_discharge_voltage, 3),
    internal_resistance = round(internal_resistance, 5),
    surface_temp = round(surface_temp, 2),
    delta_temp = round(delta_temp, 2),
    electrolyte_degradation = round(electrolyte_degradation, 4),
    soh = round(soh, 2),
    capacity_fade = round(capacity_fade, 2),
    thermal_runaway = thermal_runaway
  )
}

# Generate the primary dataset
bess_raw <- generate_bess_data(n_cells = 2800)

# Save raw data to disk for traceability
write.csv(bess_raw, "bess_tropical_raw.csv", row.names = FALSE)

# ---- 3. DATA CLEANING AND PREPROCESSING ----------------------------------------

# Check structure first. I always look for impossible values.
summary(bess_raw)

# Flag cells with SOH below 60 percent as critical outliers. In practice,
# operators retire modules well before this threshold, but the simulation lets
# a few edge cases through. We keep them in the dataset — they are physically
# plausible, just extreme. This print is purely diagnostic.
critical_outliers <- bess_raw %>% filter(soh < 60)
print(paste("Critical outlier count:", nrow(critical_outliers)))

# Remove duplicate identifiers (none expected, but robustness matters)
bess_clean <- bess_raw %>%
  distinct(cell_id, .keep_all = TRUE) %>%
  mutate(
    chemistry = factor(chemistry),
    thermal_runaway = factor(thermal_runaway, levels = c(0, 1), labels = c("No", "Yes")),
    # Create a categorical thermal stress indicator for stratified sampling
    thermal_stress = cut(ambient_temp,
                         breaks = c(0, 30, 35, 40, 100),
                         labels = c("Low", "Moderate", "High", "Severe"))
  ) %>%
  # Remove physically impossible rows (negative resistance, voltage below zero)
  filter(internal_resistance > 0,
         max_charge_voltage > 0,
         min_discharge_voltage > 0)

# Check class balance for the binary target
table(bess_clean$thermal_runaway)

# ---- 4. EXPLORATORY DATA ANALYSIS ---------------------------------------------

# Correlation matrix for numeric predictors
numeric_vars <- bess_clean %>%
  select(ambient_temp, relative_humidity, cycle_count, avg_c_rate,
         max_charge_voltage, min_discharge_voltage, internal_resistance,
         surface_temp, delta_temp, electrolyte_degradation, soh, capacity_fade)

cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Export correlation plot as PNG
png("eda_correlation_matrix.png", width = 900, height = 900, res = 120)
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix: Tropical BESS Operational Predictors",
         mar = c(0, 0, 1, 0))
dev.off()

# Distribution of SOH by chemistry
p1 <- ggplot(bess_clean, aes(x = soh, fill = chemistry)) +
  geom_density(alpha = 0.45) +
  labs(title = "SOH Distribution by Chemistry",
       x = "State of Health (%)", y = "Density") +
  theme_minimal(base_size = 11) +
  scale_fill_brewer(palette = "Set1")

# Surface temperature vs. ambient temperature, colored by TR outcome
p2 <- ggplot(bess_clean, aes(x = ambient_temp, y = surface_temp, color = thermal_runaway)) +
  geom_point(alpha = 0.35, size = 1.2) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed") +
  labs(title = "Surface Temperature vs Ambient Temperature",
       x = "Ambient Temperature (°C)", y = "Surface Temperature (°C)",
       color = "Thermal Runaway") +
  theme_minimal(base_size = 11) +
  scale_color_brewer(palette = "Dark2")

# Boxplot: capacity fade across thermal stress levels
p3 <- ggplot(bess_clean, aes(x = thermal_stress, y = capacity_fade, fill = thermal_stress)) +
  geom_boxplot(outlier.alpha = 0.3) +
  labs(title = "Capacity Fade by Thermal Stress Category",
       x = "Thermal Stress Level", y = "Capacity Fade (%)") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "YlOrRd")

# Internal resistance against cycle count, faceted by chemistry
p4 <- ggplot(bess_clean, aes(x = cycle_count, y = internal_resistance, color = chemistry)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), se = TRUE) +
  labs(title = "Internal Resistance Growth Over Cycling",
       x = "Cumulative Charge Cycles", y = "Internal Resistance (Ω)") +
  theme_minimal(base_size = 11) +
  facet_wrap(~ chemistry) +
  scale_color_brewer(palette = "Set1")

# Arrange and save
png("eda_composite_panel.png", width = 1400, height = 1000, res = 130)
grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()

# Summary statistics table by thermal runaway status
bess_clean %>%
  group_by(thermal_runaway) %>%
  summarise(
    n = n(),
    mean_soh = mean(soh),
    sd_soh = sd(soh),
    mean_surface_temp = mean(surface_temp),
    sd_surface_temp = sd(surface_temp),
    mean_internal_resistance = mean(internal_resistance),
    sd_internal_resistance = sd(internal_resistance),
    mean_cycle_count = mean(cycle_count),
    .groups = "drop"
  ) %>%
  as.data.frame()

# ---- 5. FEATURE ENGINEERING FOR MODELING --------------------------------------

# Prepare modeling matrix. Keep original physics-informed variables and
# add interaction terms that literature (Zhang et al., 2023) suggests matter.

bess_model <- bess_clean %>%
  mutate(
    temp_humidity_interaction = ambient_temp * relative_humidity / 100,
    voltage_spread = max_charge_voltage - min_discharge_voltage,
    power_stress = avg_c_rate * internal_resistance,
    log_cycle_count = log(cycle_count + 1),
    temp_above_35 = pmax(ambient_temp - 35, 0)
  )

# Define predictor sets for classification (TR) and regression (SOH)
predictors_clf <- c("chemistry", "ambient_temp", "relative_humidity", "cycle_count",
                    "avg_c_rate", "max_charge_voltage", "min_discharge_voltage",
                    "internal_resistance", "surface_temp", "delta_temp",
                    "electrolyte_degradation", "temp_humidity_interaction",
                    "voltage_spread", "power_stress", "log_cycle_count", "temp_above_35")

predictors_reg <- c("chemistry", "ambient_temp", "relative_humidity", "cycle_count",
                    "avg_c_rate", "max_charge_voltage", "min_discharge_voltage",
                    "internal_resistance", "surface_temp", "delta_temp",
                    "electrolyte_degradation", "temp_humidity_interaction",
                    "voltage_spread", "power_stress", "log_cycle_count", "temp_above_35")

# ---- 6. TRAIN-TEST SPLIT WITH STRATIFICATION ----------------------------------

# Stratified split: thermal runaway is rare (~4-6%), so stratify to preserve
# the minority class in both partitions.

train_idx <- createDataPartition(bess_model$thermal_runaway, p = 0.75, list = FALSE)
train_data <- bess_model[train_idx, ]
test_data  <- bess_model[-train_idx, ]

# Verify proportions
prop.table(table(train_data$thermal_runaway))
prop.table(table(test_data$thermal_runaway))

# ---- 7. MODEL A: RANDOM FOREST (CLASSIFICATION) -------------------------------

# Random Forest as our first non-linear ensemble. Handles mixed data types
# (factor + numeric) without explicit dummy encoding and provides a
# strong baseline for tabular classification.

rf_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = FALSE
)

rf_grid <- expand.grid(
  mtry = c(3, 5, 7, 10)
)

rf_model <- train(
  x = train_data[, predictors_clf],
  y = train_data$thermal_runaway,
  method = "rf",
  metric = "ROC",
  trControl = rf_ctrl,
  tuneGrid = rf_grid,
  ntree = 500,
  importance = TRUE
)

print(rf_model)

# Predictions on hold-out set
rf_pred_class <- predict(rf_model, newdata = test_data[, predictors_clf])
rf_pred_prob  <- predict(rf_model, newdata = test_data[, predictors_clf], type = "prob")$Yes

# Confusion matrix
rf_conf <- confusionMatrix(rf_pred_class, test_data$thermal_runaway, positive = "Yes")
print(rf_conf)

# ROC analysis
rf_roc <- roc(test_data$thermal_runaway, rf_pred_prob, levels = c("No", "Yes"), direction = "<")
cat("Random Forest AUC:", round(auc(rf_roc), 4), "\n")

# Variable importance
rf_imp <- varImp(rf_model)
print(rf_imp)

# ---- 8. MODEL B: XGBOOST (ADVANCED CLASSIFICATION) ----------------------------

# XGBoost (extreme gradient boosting) as our advanced model. Uses
# regularized objective functions, second-order Taylor approximations, and
# parallel tree construction. For tabular data with strong non-linearities and
# interactions, it generally outperforms both logistic regression and single-
# decision-tree ensembles like standard Random Forest.

# One-hot encode the categorical predictor for XGBoost
dummy_model <- dummyVars(~ chemistry, data = bess_model)
chemistry_dummies <- predict(dummy_model, newdata = bess_model)

bess_xgb <- cbind(bess_model, chemistry_dummies) %>%
  select(-chemistry, -cell_id, -thermal_stress)

# Re-split after dummy encoding
xgb_train <- bess_xgb[train_idx, ]
xgb_test  <- bess_xgb[-train_idx, ]

xgb_predictors <- setdiff(names(xgb_train), c("thermal_runaway", "soh", "capacity_fade"))

xgb_train_matrix <- xgb.DMatrix(
  data = as.matrix(xgb_train[, xgb_predictors]),
  label = ifelse(xgb_train$thermal_runaway == "Yes", 1, 0)
)

xgb_test_matrix <- xgb.DMatrix(
  data = as.matrix(xgb_test[, xgb_predictors]),
  label = ifelse(xgb_test$thermal_runaway == "Yes", 1, 0)
)

# Hyperparameter grid via cross-validation. Limited search space to keep
# runtime reasonable while still exploring key regularization knobs.

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  gamma = 0.1,
  lambda = 1.0,
  alpha = 0.5
)

xgb_cv <- xgb.cv(
  params = xgb_params,
  data = xgb_train_matrix,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 50,
  print_every_n = 100,
  prediction = TRUE,
  showsd = TRUE
)

best_nrounds <- xgb_cv$best_iteration
# Safety check: if best_iteration is 0/NULL/NA, use a sensible fallback
if (is.null(best_nrounds) || is.na(best_nrounds) || best_nrounds <= 0) {
  best_nrounds <- 100
  cat("Warning: xgb.cv best_iteration was", xgb_cv$best_iteration, 
      "— using fallback nrounds =", best_nrounds, "\n")
} else {
  cat("XGBoost best iteration:", best_nrounds, "\n")
}

# Final model
xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb_train_matrix,
  nrounds = best_nrounds,
  watchlist = list(train = xgb_train_matrix, test = xgb_test_matrix),
  print_every_n = 100
)

# Predictions
xgb_pred_prob <- predict(xgb_model, newdata = xgb_test_matrix)
xgb_pred_class <- factor(ifelse(xgb_pred_prob > 0.5, "Yes", "No"),
                          levels = c("No", "Yes"))

xgb_conf <- confusionMatrix(xgb_pred_class, xgb_test$thermal_runaway, positive = "Yes")
print(xgb_conf)

xgb_roc <- roc(xgb_test$thermal_runaway, xgb_pred_prob, levels = c("No", "Yes"), direction = "<")
cat("XGBoost AUC:", round(auc(xgb_roc), 4), "\n")

# XGBoost feature importance (gain-based)
xgb_imp <- xgb.importance(model = xgb_model)
print(xgb_imp)

# ---- 9. REGRESSION MODEL: PREDICTING SOH WITH XGBOOST -------------------------

# Same XGBoost framework applied to the continuous SOH target. Treats
# capacity degradation as a regression surface rather than a class label,
# which is more appropriate for maintenance scheduling.

soh_train_matrix <- xgb.DMatrix(
  data = as.matrix(xgb_train[, xgb_predictors]),
  label = xgb_train$soh
)

soh_test_matrix <- xgb.DMatrix(
  data = as.matrix(xgb_test[, xgb_predictors]),
  label = xgb_test$soh
)

soh_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 4,
  eta = 0.05,
  subsample = 0.85,
  colsample_bytree = 0.85,
  min_child_weight = 2,
  gamma = 0.05,
  lambda = 1.0
)

soh_cv <- xgb.cv(
  params = soh_params,
  data = soh_train_matrix,
  nrounds = 800,
  nfold = 5,
  early_stopping_rounds = 50,
  print_every_n = 100
)

soh_best_nrounds <- soh_cv$best_iteration
# Safety check: if best_iteration is 0/NULL/NA, use a sensible fallback
if (is.null(soh_best_nrounds) || is.na(soh_best_nrounds) || soh_best_nrounds <= 0) {
  soh_best_nrounds <- 80
  cat("Warning: soh_cv best_iteration was", soh_cv$best_iteration, 
      "— using fallback nrounds =", soh_best_nrounds, "\n")
}

soh_model <- xgb.train(
  params = soh_params,
  data = soh_train_matrix,
  nrounds = soh_best_nrounds
)

soh_pred <- predict(soh_model, newdata = soh_test_matrix)
soh_rmse <- sqrt(mean((soh_pred - xgb_test$soh)^2))
soh_mae  <- mean(abs(soh_pred - xgb_test$soh))
soh_r2   <- 1 - sum((soh_pred - xgb_test$soh)^2) / sum((xgb_test$soh - mean(xgb_test$soh))^2)

cat("SOH Regression — RMSE:", round(soh_rmse, 4),
    "MAE:", round(soh_mae, 4),
    "R-squared:", round(soh_r2, 4), "\n")

# ---- 10. MODEL COMPARISON AND VISUALIZATION -----------------------------------

# ROC curves overlay
png("model_comparison_roc.png", width = 900, height = 700, res = 120)
plot(rf_roc, col = "darkgreen", lwd = 2, main = "ROC Comparison: Random Forest vs XGBoost")
plot(xgb_roc, col = "darkblue", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(rf_roc), 3)),
                  paste0("XGBoost (AUC = ", round(auc(xgb_roc), 3))),
       col = c("darkgreen", "darkblue"), lwd = 2)
dev.off()

# Feature importance side-by-side (top 10 features)
# Random Forest importance from caret::varImp — the column name varies, so we extract the first numeric column
rf_imp_df <- rf_imp$importance %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  rename(Importance = 2) %>%  # rename the first (and usually only) numeric column
  mutate(Overall = Importance / max(Importance)) %>%
  arrange(desc(Overall)) %>%
  head(10) %>%
  mutate(Model = "Random Forest") %>%
  select(Feature, Overall, Model)

xgb_imp_df <- xgb_imp %>%
  select(Feature, Gain) %>%
  mutate(Overall = Gain / max(Gain), Model = "XGBoost") %>%
  arrange(desc(Overall)) %>%
  head(10) %>%
  select(Feature, Overall, Model)

importance_combined <- bind_rows(rf_imp_df, xgb_imp_df)

p_imp <- ggplot(importance_combined, aes(x = reorder(Feature, Overall), y = Overall, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Normalized Feature Importance: Top 10 Predictors",
       x = "Feature", y = "Normalized Importance") +
  theme_minimal(base_size = 11) +
  scale_fill_brewer(palette = "Set1")

png("feature_importance_comparison.png", width = 900, height = 700, res = 120)
print(p_imp)
dev.off()

# SOH prediction vs actual
soh_df <- tibble(Actual = xgb_test$soh, Predicted = soh_pred)

p_soh <- ggplot(soh_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  geom_smooth(method = "lm", color = "darkgreen", se = FALSE) +
  labs(title = "SOH Prediction: Actual vs Predicted (XGBoost Regression)",
       x = "Actual SOH (%)", y = "Predicted SOH (%)") +
  annotate("text", x = 80, y = 65, label = paste0("R² = ", round(soh_r2, 3)),
           color = "darkred", size = 4.5) +
  theme_minimal(base_size = 11)

png("soh_regression_scatter.png", width = 800, height = 700, res = 120)
print(p_soh)
dev.off()

# ---- 11. BUSINESS RULES FOR OPERATIONAL STRATEGY ------------------------------

# Tiered alert framework that a plant operator could embed into a SCADA dashboard.

bess_scored <- test_data %>%
  mutate(
    tr_probability = xgb_pred_prob[seq_len(nrow(test_data))],
    predicted_soh = soh_pred[seq_len(nrow(test_data))],
    # Tiered alert logic
    alert_level = case_when(
      tr_probability > 0.35 | predicted_soh < 75 ~ "CRITICAL",
      tr_probability > 0.15 | predicted_soh < 82 ~ "WARNING",
      tr_probability > 0.05 | predicted_soh < 88 ~ "WATCH",
      TRUE ~ "NORMAL"
    )
  )

alert_summary <- bess_scored %>%
  group_by(alert_level) %>%
  summarise(
    count = n(),
    pct = round(100 * n() / nrow(bess_scored), 1),
    avg_tr_prob = round(mean(tr_probability), 4),
    avg_soh = round(mean(predicted_soh), 2),
    .groups = "drop"
  )

print(alert_summary)

# Save scored test set for downstream reporting
write.csv(bess_scored, "bess_scored_test_set.csv", row.names = FALSE)

# ---- 12. SESSION INFO FOR REPRODUCIBILITY -------------------------------------

sink("session_info.txt")
cat("=== Session Information ===\n")
print(sessionInfo())
cat("\n=== Reproducibility Seed ===\n")
cat("Seed: 202406\n")
cat("\n=== Dataset Dimensions ===\n")
cat("Raw observations:", nrow(bess_raw), "\n")
cat("Clean observations:", nrow(bess_clean), "\n")
cat("Training set:", nrow(train_data), "\n")
cat("Test set:", nrow(test_data), "\n")
sink()

# =============================================================================
# END OF SCRIPT
# =============================================================================
