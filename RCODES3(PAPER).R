#-------------------------------------------------------------------------------------------------------
# R(version 4.4.0)
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
#Required R packages
#-------------------------------------------------------------------------------------------------------

library(tidyverse)
library(forecast)
library(tseries)
library(ggplot2)
library(moments)
library(zoo)
library(FinTS)
library(TSA)
library(data.table)
library(reshape2)
library(GAS)
library(rugarch)
library(xgboost) 
library(PerformanceAnalytics)


#Importing Data fromcsv excel
Top40_data_Analysis <- read.csv("C:/Users/israel.maingo/Downloads/Top40_data_Analysis.csv",
                                sep = ",",
                                fileEncoding = "UTF-8",
                                stringsAsFactors = FALSE)
#Arrange the Data in a proper Date format
Top40_data_Analysis$Date <- as.Date(Top40_data_Analysis$Date, format = "%Y/%m/%d")
head(Top40_data_Analysis)

#Coverting Data into time series(ts())
set.seed(7553) 
z3=ts(Top40_data_Analysis$Price,start=c(2011), frequency=260) 
z3


#----------------------------------------------------------------------------------
#Differencing to get Log-Returns
#------------------------------------------------------------------------

set.seed(6342)
y3 <-log(z3)
y3
log_returns3 <- diff (log(z3) )*100
log_returns3


t=ts(log_returns3 , start =c (2011), frequency =260)



#---------------------------------------------------------------------------
#Time series plots,QQ, and Time series Decomposition
#---------------------------------------------------------------------------

win.graph ()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par ( mfrow =c(2 ,2) )
w =ts(y3 , start =c (2011) , frequency =260)
par( cex.main = 2 , cex.axis = 1.6 , cex.lab = 1.5 , pch = 20)
plot (t, xlab =" Year ", ylab ="Log of JSE Top40 Index ",col ="blue",
      main ="(a) JSE Top40 Index ")
grid(col = "white", lty = "solid", lwd = 1)
plot (t, xlab =" Year ", ylab =" Daily log - returns ",col ="blue",
      main ="(b) Daily log - returns of JSE Top40 Index ")
grid(col = "white", lty = "solid", lwd = 1)
plot ( density (t) , xlab =" Daily log - returns ", main ="(c) Probability density
",
       col = "blue")
grid(col = "white", lty = "solid", lwd = 1)
qqnorm (t,col = "blue", main ="(d) Normal QQ plot ")
qqline (t,col = "red")
grid(col = "white", lty = "solid", lwd = 1)


#Time Series Decomposition 
win.graph ()
set.seed (6168)
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par( cex.main = 2 , cex.axis = 1.6 , cex.lab = 1.5 , pch = 20)
fit.stl <- stl(t, t.window =12 , s.window ="periodic", robust = TRUE)
fit.stl
plot( fit.stl ,col ="darkred", main ="Time series decomposition of JSE Top40 log-returns")
grid(col = "white", lty = "solid", lwd = 1)


#------------------------------------------------------------------------------------
#Fitting auto.arima()-Mean function of the Log-Returns (To select the best mean model)
#--------------------------------------------------------------------------------------

auto_arima <- forecast::auto.arima(t,d = 0, max.p = 10, max.q = 10, 
                                   seasonal = FALSE, stepwise = FALSE, 
                                   approximation = FALSE)
auto_arima



#----------------------------------------------------------------------------------------------------------------------------
# Fitting the XGBoost from the Residuals Extracted from ARMA(3,2)-EGARCH(1,1)
#----------------------------------------------------------------------------------------------------------------------------

# STEP 1: Fit the ARMA(3,2)-EGARCH(1,1) model to data 't'
spec_egarch <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
  mean.model     = list(armaOrder = c(3,2), include.mean = TRUE),
  distribution.model = "sstd"  # "std", "sstd", "ged", "sged", "ghyp"
)

fit_egarch <- rugarch::ugarchfit(spec = spec_egarch, data = t)
fit_egarch  # Display the model fit results

# STEP 2: Extract standardized residuals
std_resid <- fit_egarch@fit$z
std_resid <- as.numeric(std_resid)

# Convert to data.table
resid_dt <- data.table(std_resid = std_resid)

# STEP 3: Create lag and rolling features
for (i in 1:15) {
  resid_dt[, paste0("lag", i) := shift(std_resid, i)]
}

for (k in c(2, 3, 5, 10, 20)) {
  resid_dt[, paste0("roll_mean", k) := rollmean(std_resid, k, fill = NA, align = "right")]
  resid_dt[, paste0("roll_sd", k) := rollapply(std_resid, width = k, FUN = sd, fill = NA, align = "right")]
}

resid_dt <- na.omit(resid_dt)

#----------------------------------------------------------------------------------------------------------------------------
# UPDATED SECTION: Split Data (60% Train, 20% Calibration, 20% Test)
#----------------------------------------------------------------------------------------------------------------------------
set.seed(548)
n <- nrow(resid_dt)
train_end <- floor(0.6 * n)
calib_end <- floor(0.8 * n)

train_data <- resid_dt[1:train_end]
calib_data <- resid_dt[(train_end + 1):calib_end]
test_data  <- resid_dt[(calib_end + 1):n]

x_train <- as.matrix(train_data[, !"std_resid"])
y_train <- as.vector(train_data$std_resid)

x_calib <- as.matrix(calib_data[, !"std_resid"])
y_calib <- as.vector(calib_data$std_resid)

x_test <- as.matrix(test_data[, !"std_resid"])
y_test <- as.vector(test_data$std_resid)

cat("Data Split Sizes:\n")
cat("Training:", nrow(train_data), "\n")
cat("Calibration:", nrow(calib_data), "\n")
cat("Testing:", nrow(test_data), "\n\n")

#----------------------------------------------------------------------------------------------------------------------------
# STEP 4: Train the hybrid ARMA(3,2)-EGARCH(1,1)-XGBoost model
#----------------------------------------------------------------------------------------------------------------------------
dtrain_all <- xgb.DMatrix(data = x_train, label = y_train)

cv_model <- xgb.cv(
  data = dtrain_all,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 20,
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

best_nrounds <- cv_model$best_iteration
cat("Best number of boosting rounds:", best_nrounds, "\n")

xgb_model_final <- xgboost(
  data = dtrain_all,
  nrounds = best_nrounds,
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

#----------------------------------------------------------------------------------------------------------------------------
# STEP 5: Prediction and Calibration for PI
#----------------------------------------------------------------------------------------------------------------------------
y_pred_calib <- predict(xgb_model_final, newdata = x_calib)
resid_calib <- y_calib - y_pred_calib

# Compute quantiles for 90%, 95%, and 99% prediction intervals
lower_qs <- quantile(resid_calib, probs = c(0.05, 0.025, 0.005))
upper_qs <- quantile(resid_calib, probs = c(0.95, 0.975, 0.995))

#----------------------------------------------------------------------------------------------------------------------------
# STEP 6: Predict on Test and Construct PIs
#----------------------------------------------------------------------------------------------------------------------------
y_pred_test <- predict(xgb_model_final, newdata = x_test)


# Create test_index for plotting
test_index <- (calib_end + 1):n 

lower_90 <- y_pred_test + lower_qs[1]
upper_90 <- y_pred_test + upper_qs[1]
lower_95 <- y_pred_test + lower_qs[2]
upper_95 <- y_pred_test + upper_qs[2]
lower_99 <- y_pred_test + lower_qs[3]
upper_99 <- y_pred_test + upper_qs[3]

results <- data.table(
  actual = y_test,
  prediction = y_pred_test,
  lower_90 = lower_90,
  upper_90 = upper_90,
  lower_95 = lower_95,
  upper_95 = upper_95,
  lower_99 = lower_99,
  upper_99 = upper_99
)

results[, `:=`(
  width_90 = upper_90 - lower_90,
  width_95 = upper_95 - lower_95,
  width_99 = upper_99 - lower_99
)]

results_to_10_predict <- head(results, 10)
print(results_to_10_predict)

#----------------------------------------------------------------------------------------------------------------------------
# STEP 7: Forecast Accuracy Metrics
#----------------------------------------------------------------------------------------------------------------------------
mse   <- mean((results$prediction - results$actual)^2)
rmse  <- sqrt(mse)
mae   <- mean(abs(results$prediction - results$actual))
mape  <- mean(abs((results$prediction - results$actual) / results$actual)) * 100
smape <- mean(2 * abs(results$prediction - results$actual) /
                (abs(results$prediction) + abs(results$actual))) * 100

naive_forecast <- results$actual[-length(results$actual)]
actual_trimmed <- results$actual[-1]
mae_naive <- mean(abs(actual_trimmed - naive_forecast))
mase <- mae / mae_naive

cat("Forecast Performance:\n")
cat("MSE   =", round(mse, 4), "\n")
cat("RMSE  =", round(rmse, 4), "\n")
cat("MAE   =", round(mae, 4), "\n")
cat("MAPE  =", round(mape, 4), "%\n")
cat("sMAPE =", round(smape, 4), "%\n")
cat("MASE  =", round(mase, 4), "\n\n")


# -------------------------------
#  required packages
# -------------------------------
library(GAS)
library(ggplot2)
library(reshape2)

# --------------------------------------------------------------------------------------
# Fitting GAS models under conditional distributions: std, sstd, norm,snorm,ast,ast1,& ald
# ---------------------------------------------------------------------------------------

GAS_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", 
                       GASPar = list(location = TRUE, scale = TRUE, skewness = FALSE, shape = TRUE)) # Time-varying parameters

GAS_fit <- UniGASFit(GAS_spec, t)
print(GAS_fit)


library(GAS)
#iid test and Histogram of pit
U = pit(GAS_fit)
Test = PIT_test(U, G = 20, alpha = 0.05, plot = TRUE)


# -----------------------------------------------------------------------------
# Rolling forecast (Out-of-sample forecast)
# ---------------------------------------------------------------------------

Roll <- UniGASRoll(t, GAS_spec,
                   ForecastLength = 250,  # length of rolling forecasts
                   RefitEvery = 10,
                   RefitWindow = "moving")
Roll

# ---------------------------------------------------------------------------------
#  Extracting point forecasts
# ----------------------------------------------------------------------------------
forecast_df <- as.data.frame(Roll@Forecast$PointForecast)
forecast_df$Time <- 1:nrow(forecast_df)

# Actual returns and realised volatility
actual_returns <- t[(length(t) - nrow(forecast_df) + 1):length(t)]
realised_vol <- abs(actual_returns)

forecast_df$ActualReturns <- actual_returns
forecast_df$RealisedVol <- realised_vol

# ----------------------------------------------------------------------------------------------
#  Plot Conditional Location (alone)
# ----------------------------------------------------------------------------------------------
ggplot(forecast_df, aes(x = Time, y = location)) +
  geom_line(color = "blue", size = 1.2) +
  labs(title = "Conditional Location (Mean)",
       x = "Time", y = "Location") +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "lightgray", color = NA),
    panel.background = element_rect(fill = "lightgray", color = NA),
    panel.grid.major = element_line(color = "white", size = 1),
    panel.grid.minor = element_line(color = "white", size = 0.5),
    axis.line = element_line(color = "black", size = 0.8),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 13),
    plot.title = element_text(size = 16, hjust = 0.5)
  )

# ---------------------------------------------------------------------------------------------------
#  Plot Conditional Scale (alone)
# ---------------------------------------------------------------------------------------------------
ggplot(forecast_df, aes(x = Time, y = scale)) +
  geom_line(color = "red", size = 1.2) +
  labs(title = "Conditional Scale (Volatility)",
       x = "Time", y = "Scale") +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "lightgray", color = NA),
    panel.background = element_rect(fill = "lightgray", color = NA),
    panel.grid.major = element_line(color = "white", size = 1),
    panel.grid.minor = element_line(color = "white", size = 0.5),
    axis.line = element_line(color = "black", size = 0.8),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 13),
    plot.title = element_text(size = 16, hjust = 0.5)
  )

# ----------------------------------------------------------------------------------
#  Plot Conditional Shape (alone)
# -----------------------------------------------------------------------------------
ggplot(forecast_df, aes(x = Time, y = shape)) +
  geom_line(color = "green4", size = 1.2) +
  labs(title = "Conditional Shape Parameter",
       x = "Time", y = "Shape") +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "lightgray", color = NA),
    panel.background = element_rect(fill = "lightgray", color = NA),
    panel.grid.major = element_line(color = "white", size = 1),
    panel.grid.minor = element_line(color = "white", size = 0.5),
    axis.line = element_line(color = "black", size = 0.8),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 13),
    plot.title = element_text(size = 16, hjust = 0.5)
  )


# -------------------------------------------------------------------------------
#  Plot Scale vs Realised Volatility
# -------------------------------------------------------------------------------
ggplot(forecast_df, aes(x = Time)) +
  geom_line(aes(y = RealisedVol), color = "black", size = 1) +
  geom_line(aes(y = scale), color = "red", size = 1.2) +
  labs(title = "Scale vs Realised Volatility",
       x = "Time", y = "Volatility") +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "lightgray", color = NA),
    panel.background = element_rect(fill = "lightgray", color = NA),
    panel.grid.major = element_line(color = "white", size = 1),
    panel.grid.minor = element_line(color = "white", size = 0.5),
    axis.line = element_line(color = "black", size = 0.8),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 13),
    plot.title = element_text(size = 16, hjust = 0.5)
  )


InSampleData = t[1:2812] 
OutSampleData = t[2813:3515] 
Fit = UniGASFit(GAS_spec, InSampleData) 
Forecast = UniGASFor(Fit, Roll = TRUE, out = OutSampleData) 
alpha = 0.05 
VaR = quantile(Forecast, alpha) 
BackTest = BacktestVaR(OutSampleData, VaR, alpha) 
BackTest


alpha <- 0.05  # 5% VaR
mu <- forecast_df$location
sigma <- forecast_df$scale
nu <- forecast_df$shape  # degrees of freedom

# VaR (5%)
forecast_df$VaR_5 <- mu + sigma * qt(alpha, df = nu)  # Student-t quantile

# ES (5%) using formula for Student-t
pdf_t <- function(x, df) dt(x, df = df)
forecast_df$ES_5 <- mu - sigma * ((nu + qt(alpha, df=nu)^2) / (nu - 1)) * pdf_t(qt(alpha, df=nu), df=nu) / alpha

# Combined VaR and ES plot
win.graph()
ggplot(forecast_df, aes(x = Time)) +
  geom_line(aes(y = VaR_5, color = "VaR (5%)"), size = 1.2) +
  geom_line(aes(y = ES_5, color = "ES (5%)"), size = 1.2) +
  labs(title = "Rolling 5% VaR and ES",
       x = "Forecast Horizon", y = "Risk Measure") +
  scale_color_manual(values = c("VaR (5%)" = "darkgreen", "ES (5%)" = "purple")) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "lightgray", color = NA),
    panel.background = element_rect(fill = "lightgray", color = NA),
    panel.grid.major = element_line(color = "white", size = 1),
    panel.grid.minor = element_line(color = "white", size = 0.5),
    axis.line = element_line(color = "black", size = 0.8),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 13),
    plot.title = element_text(size = 16, hjust = 0.5),
    legend.title = element_blank(),
    legend.position = "top"
  )

head_forecasts <- head(forecast_df[, c("Time", "VaR_5", "ES_5")], 10)
print(head_forecasts)


#------------------------------------------------------------------------------------------------------
#Perform rolling forecasts (out-of-sample forecasts) for ARMA3,2)-EGARCH(1,1)
#------------------------------------------------------------------------------------------------------

roll_egarch <- ugarchroll(
  spec = spec_egarch,
  data = t,
  n.ahead = 1,         # Forecast horizon (1-step ahead)
  forecast.length = 250,  # Number of rolling forecasts
  refit.every = 10,    # Refit frequency
  refit.window = "moving", # Moving or expanding window
  solver = "hybrid",
  fit.control = list(stationarity = 1)
)

#Extracting forecasts
roll_results <- as.data.frame(roll_egarch)

head(roll_results)


#-------------------------------------------------------------------------------------------
# DM Test and MCS Procedure (Updated Full DM Table)
#----------------------------------------------------------------------------------------------
library(forecast)
library(MCS)

#---------------------------------------------------------------------------
# Compute Losses
#-------------------------------------------------------------------------
# Assume equal-length forecast error series
loss_hybrid <- (results$actual - results$prediction)^2
loss_egarch <- (roll_results$Realized - roll_results$Sigma)^2
loss_gas    <- (forecast_df$RealisedVol - sigma)^2

# Combine into loss matrix
loss_matrix <- cbind(loss_hybrid, loss_egarch, loss_gas)
colnames(loss_matrix) <- c("Hybrid", "ARMA-EGARCH", "GAS-STD")

#------------------------------------------------------------------
# Pairwise DM Tests (Full Table)
#------------------------------------------------------------------
n_models <- ncol(loss_matrix)
dm_p_full <- matrix(NA, nrow = n_models, ncol = n_models)
colnames(dm_p_full) <- rownames(dm_p_full) <- colnames(loss_matrix)

set.seed(6879)
for(i in 1:n_models){
  for(j in 1:n_models){
    if(i != j){
      dm_p_full[i, j] <- dm.test(loss_matrix[, i], loss_matrix[, j])$p.value
    } else {
      dm_p_full[i, j] <- NA  # diagonal
    }
  }
}

cat("Full DM Test P-Value Matrix:\n")
print(round(dm_p_full, 6))

#--------------------------------------------------------------------------------------------
# MCS Procedure
#---------------------------------------------------------------------------------------------
mcs_res <- MCSprocedure(Loss = loss_matrix, alpha = 0.05, B = 1000, statistic = "Tmax")

cat("\nMCS Superior Set Results:\n")
print(mcs_res)

cat("\nSummary of MCS:\n")
summary(mcs_res)


#-----------------------------------------------------------------------------------------------
#RMSE Bootstrap CI and DM
#-----------------------------------------------------------------------------------------------

library(boot)
# Hybrid model
loss_hybrid <- (results$actual - results$prediction)^2

# ARMA-EGARCH model
loss_egarch <- (roll_results$Realized - roll_results$Sigma)^2

# GAS-STD model
loss_gas <- (forecast_df$RealisedVol - sigma)^2

# Combine into a matrix
loss_matrix <- cbind(Hybrid = loss_hybrid,
                     ARMA_EGARCH = loss_egarch,
                     GAS_STD = loss_gas)

# RMSE function
rmse_func <- function(data, indices) {
  actual <- data[indices, 1]  # column 1 is actual
  predicted <- data[indices, 2] # column 2 is prediction
  sqrt(mean((actual - predicted)^2))
}

# Prepare actuals and predictions for each model
rmse_bootstrap <- function(actual, predicted, R=1000) {
  data <- cbind(actual, predicted)
  boot_res <- boot(data = data, statistic = rmse_func, R = R)
  ci <- boot.ci(boot_res, type = "perc")$percent[4:5]
  point <- sqrt(mean((actual - predicted)^2))
  return(list(RMSE = point, CI_lower = ci[1], CI_upper = ci[2]))
}

# Compute for each model
rmse_hybrid <- rmse_bootstrap(results$actual, results$prediction)
rmse_egarch <- rmse_bootstrap(roll_results$Realized, roll_results$Sigma)
rmse_gas <- rmse_bootstrap(forecast_df$RealisedVol, sigma)

# Combine results
rmse_results <- data.frame(
  Model = c("Hybrid", "ARMA(3,2)-EGARCH1,1)", "GAS-STD"),
  RMSE = c(rmse_hybrid$RMSE, rmse_egarch$RMSE, rmse_gas$RMSE),
  CI_Lower = c(rmse_hybrid$CI_lower, rmse_egarch$CI_lower, rmse_gas$CI_lower),
  CI_Upper = c(rmse_hybrid$CI_upper, rmse_egarch$CI_upper, rmse_gas$CI_upper)
)

print(rmse_results)

# Function for bootstrap DM statistic
dm_boot_func <- function(data, indices) {
  e1 <- data[indices,1]
  e2 <- data[indices,2]
  mean(e1 - e2) / sd(e1 - e2) * sqrt(length(e1))
}

# DM test Hybrid vs ARMA(3,2)-EGARCH(1,1)
set.seed(123)
dm_hybrid_egarch <- boot(cbind(loss_hybrid, loss_egarch), statistic = dm_boot_func, R = 1000)
dm_hybrid_gas <- boot(cbind(loss_hybrid, loss_gas), statistic = dm_boot_func, R = 1000)
dm_egarch_gas <- boot(cbind(loss_egarch, loss_gas), statistic = dm_boot_func, R = 1000)

# Extract 95% CI
dm_ci <- function(dm_boot) boot.ci(dm_boot, type = "perc")$percent[4:5]

data.frame(
  Comparison = c("Hybrid vs ARMA-EGARCH", "Hybrid vs GAS-STD", "ARMA-EGARCH vs GAS-STD"),
  DM_CI_Lower = c(dm_ci(dm_hybrid_egarch)[1], dm_ci(dm_hybrid_gas)[1], dm_ci(dm_egarch_gas)[1]),
  DM_CI_Upper = c(dm_ci(dm_hybrid_egarch)[2], dm_ci(dm_hybrid_gas)[2], dm_ci(dm_egarch_gas)[2])
)


#---------------------------------------------------------------------------------------------
#fit GAS model with different scaling
#---------------------------------------------------------------------------------------------

GAS_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", 
                       GASPar = list(location = TRUE, scale = TRUE, skewness = FALSE, shape = TRUE)) # Time-varying parameters

fit_identity  <- UniGASFit(GAS_spec, t)
print(fit_identity )



GAS_spec1 <- UniGASSpec(Dist = "std", ScalingType = "InvFisher", 
                        GASPar = list(location = TRUE, scale = TRUE, skewness = FALSE, shape = TRUE)) # Time-varying parameters


fit_invFisher <- UniGASFit(GAS_spec1, t)
print(fit_invFisher)



#----------------------------------------------------------------------------------
# SIMULATIONS 
#-----------------------------------------------------------------------------------

A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331,5)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std", 
                 ScalingType = "Identity")
Sim
sim_series <- as.numeric(Sim@Data[[1]]) # flatten the matrix to a vector
sim_series
library(moments)
kurt <- kurtosis(sim_series)
kurt


A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331, 8)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim1 <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std", 
                  ScalingType = "Identity")
Sim1
sim_series1 <- as.numeric(Sim1@Data[[1]])
library(moments)
kurt1 <- kurtosis(sim_series1)
kurt1

A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331, 10)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim2 <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std", 
                  ScalingType = "Identity")
Sim2
sim_series2 <- as.numeric(Sim2@Data[[1]])
library(moments)
kurt2 <- kurtosis(sim_series2)
kurt2

A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331, 15)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim3 <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std",
                  ScalingType = "Identity")
Sim3
sim_series3 <- as.numeric(Sim3@Data[[1]])
library(moments)
kurt3 <- kurtosis(sim_series3)
kurt3



A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331, 20)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim4 <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std", 
                  ScalingType = "Identity")
Sim4
sim_series4 <- as.numeric(Sim4@Data[[1]])
library(moments)
kurt4 <- kurtosis(sim_series4)
kurt4



A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331, 25)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim5 <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std",
                  ScalingType = "Identity")
Sim5
sim_series5 <- as.numeric(Sim5@Data[[1]])
library(moments)
kurt5 <- kurtosis(sim_series5)
kurt5

A <- diag(c(0.03574515, -0.003421556, 0.0))
B <- diag(c(0.0000009998613, 0.9763652, 0.0))
ThetaStar <- c(0.05472993, 0.86522331, 30)
kappa <- (diag(3) - B) %*% UniUnmapParameters(ThetaStar, "std")
set.seed(2345)
Sim6 <- UniGASSim(T.sim = 10000, kappa = kappa, A = A,, B = B, Dist = "std", 
                  ScalingType = "Identity")
Sim6
sim_series6 <- as.numeric(Sim6@Data[[1]])
library(moments)
set.seed(2345)
kurt6 <- kurtosis(sim_series6)
kurt6

#-----------------------------------------------------------------------
#    QQ-plots
#------------------------------------------------------------------------
win.graph(width = 10)
par(mfrow=c(2,4))
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 1.4, cex.axis = 1.6, cex.lab = 1.5, pch = 16)
qqnorm(sim_series, main = "QQ plot with v = 5")
qqline(sim_series, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)
qqnorm(sim_series1, main = "QQ plot with v = 8")
qqline(sim_series1, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)
qqnorm(sim_series2, main = "QQ plot with v = 10")
qqline(sim_series2, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)
qqnorm(sim_series3, main = "QQ plot with v = 15")
qqline(sim_series3, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)
qqnorm(sim_series4, main = "QQ plot with v = 20")
qqline(sim_series4, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)
qqnorm(sim_series5, main = "QQ plot with v = 25")
qqline(sim_series5, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)
qqnorm(sim_series6, main = "QQ plot with v = 30")
qqline(sim_series6, col="red")
# Add grid lines with thicker white grid lines
grid(col = "white", lty = 1, lwd = 2)

#------------------------------------------------------------------------------
#VaR/ES 5% and Plots of the Simulated Data
#-------------------------------------------------------------------------------

#VaR and ES 5% 
alpha <- 0.05 # 5% VaR 
VaR_5 <- quantile(sim_series, probs = alpha) 
VaR_5 
ES_5 <- mean(sim_series[sim_series<= VaR_5]) # 5% ES 
ES_5


#Time series and Histogram plots
win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 1.4, cex.axis = 1.6, cex.lab = 1.5, pch = 16)
plot(sim_series, type="l", main="Time series plot of Simulated data with 5% VaR", ylab="Log Returns", xlab="Index") 
points(which(sim_series <= VaR_5), sim_series[sim_series <= VaR_5], col="red", pch=16)
grid(col = "white", lty = 1, lwd = 2)

win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 1.4, cex.axis = 1.6, cex.lab = 1.5, pch = 16)
hist(sim_series, breaks=50, col="lightblue", main="Histogram of Simulated data with 5% VaR and ES", xlab="Log Returns")
abline(v=VaR_5, col="red", lwd=2, lty=2)
abline(v=ES_5, col="darkred", lwd=2, lty=2)
hist(sim_series[sim_series <= VaR_5], breaks=10, col=rgb(1,0,0,0.5), add=TRUE) # Highlight tail
legend("topright", legend=c("VaR 5%", "ES 5%"), col=c("red","darkred"), lty=2, lwd=2)
grid(col = "white", lty = 1, lwd = 2)





#--------------------------------------------------------------------------------------------------
#Comaparison Analysis of JSE Top40 Index and MOEX Index
#---------------------------------------------------------------------------------------------------
library(PerformanceAnalytics)


Top40data1 <- read.csv("C:/Users/israel.maingo/Downloads/Top40DataAnalysis.csv",
                       sep = ",",
                       fileEncoding = "UTF-8",
                       stringsAsFactors = FALSE)
Top40data1$Date <- as.Date(Top40data1$Date, format = "%Y/%m/%d")
Top40data2 <- Top40data1[order(Top40data1$Date), ]
Top40data2 



MOEXdata <- read.csv("C:/Users/israel.maingo/Downloads/MOEXData.csv", 
                     sep = ";", fileEncoding = "UTF-8", 
                     stringsAsFactors = FALSE, 
                     na.strings = c("", "NA") ) 

MOEXdata$Date <- as.Date(MOEXdata$Date, format = "%d-%m-%Y")
MOEXdata

set.seed(5375) 
z1=ts(Top40data2$Price,start=c(2011), frequency=260) 
z1 

set.seed(5778) 
z2=ts(MOEXdata$Price,start=c(2011), frequency=260) 
z2



set.seed(4563)
y1 <-log(z1)
y1
log_returns1 <- diff (log(z1) )*100
log_returns1


set.seed(4765)
y2 <-log(z2)
y2
log_returns2 <- diff (log(z2) )*100
log_returns2


t1=ts(log_returns1 , start =c (2011) , frequency =260)
t2=ts(log_returns2 , start =c (2011) , frequency =260)


#Log-Returns Plots
win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 2, cex.axis = 1.4, cex.lab = 1.4, pch = 50)
plot(log_returns1,
     type = "l",
     col = "blue",
     lwd = 1.2,
     ylab = "Log-Returns",
     xlab = "Year",
     main = "Time Series Plot of Log-Returns: JSE Top40 vs MOEX Index")

lines(log_returns2,
      col = "red",
      lwd = 1.2)

legend("topleft",
       legend = c("JSE Top40 log-returns", "MOEX log-returns"),
       col = c("blue", "red"),
       lty = 1,
       lwd = 1.2,
       bty = "n")
grid(col = "white", lty = "solid", lwd = 1)

stats <- data.frame(
  Index = c("JSE Top40", "MOEX"),
  Mean = c(mean(log_returns1), mean(log_returns2)),
  SD = c(sd(log_returns1), sd(log_returns2)),
  Min = c(min(log_returns1), min(log_returns2)),
  Max = c(max(log_returns1), max(log_returns2)),
  Skewness = c(skewness(log_returns1), skewness(log_returns2)),
  Kurtosis = c(kurtosis(log_returns1), kurtosis(log_returns2))
)
print(stats)

log_returns3<-head(log_returns2,3702)

win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 2, cex.axis = 1.4, cex.lab = 1.4, pch = 50)
plot(log_returns1, log_returns3,
     xlab = "JSE Top40 log-returns",
     ylab = "MOEX log-returns",
     main = "Scatter Plot of Log-Returns: JSE vs MOEX",
     col = "darkgreen", pch = 16)
grid(col = "white", lty = "solid", lwd = 1)

roll_sd_jse <- rollapply(log_returns1, width = 20, FUN = sd, fill = NA)
roll_sd_moex <- rollapply(log_returns2, width = 20, FUN = sd, fill = NA)

win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 2, cex.axis = 1.4, cex.lab = 1.4, pch = 50)
plot(roll_sd_jse, type = "l", col = "blue", lwd = 2,
     ylab = "Rolling Volatility", xlab = "Year",
     main = "20-day Rolling Volatility: JSE vs MOEX")
lines(roll_sd_moex, col = "red", lwd = 2)
legend("topleft", legend = c("JSE Top40", "MOEX"), col = c("blue", "red"), lty = 1, lwd = 2)
grid(col = "white", lty = "solid", lwd = 1)


start_year <- 2011
end_year <- 2025
n <- length(log_returns1)


years_seq <- seq(from = start_year, to = end_year, length.out = n)


cum_returns_jse <- cumsum(log_returns1)
cum_returns_moex <- cumsum(log_returns3)


win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 2, cex.axis = 1.4, cex.lab = 1.4, pch = 50)
plot(years_seq, cum_returns_jse, type = "l", col = "blue", lwd = 2,
     ylab = "Cumulative Log-Returns", xlab = "Year",
     main = "Cumulative Log-Returns: JSE vs MOEX")
lines(years_seq, cum_returns_moex, col = "red", lwd = 2)
legend("topleft", legend = c("JSE Top40", "MOEX"), col = c("blue", "red"), lty = 1, lwd = 2)
grid(col = "white", lty = "solid", lwd = 1)


GAS_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", 
                       GASPar = list(location = TRUE, scale = TRUE, skewness = FALSE, shape = TRUE)) # Time-varying parameters

GAS_fit <- UniGASFit(GAS_spec, t1)
print(GAS_fit)




std_resids <- residuals(GAS_fit, standardize = TRUE)
qqnorm(std_resids)
qqline(std_resids, col="red")



Roll_out <- UniGASRoll(t1, GAS_spec, ForecastLength = 250,
                       RefitEvery = 10, RefitWindow = "moving")
Roll1_out

InSampleData = t1[1:2962] 
OutSampleData = t1[2962:3702] 
Fit = UniGASFit(GAS_spec, InSampleData) 
Forecast = UniGASFor(Fit, Roll_out = TRUE, out = OutSampleData) 
alpha = 0.05 
VaR = quantile(Forecast, alpha) 
BackTest = BacktestVaR(OutSampleData, VaR, alpha) 
BackTest

GAS_spec1 <- UniGASSpec(Dist = "std", ScalingType = "Identity", 
                        GASPar = list(location = TRUE, scale = TRUE, skewness = FALSE, shape = FALSE)) # Time-varying parameters

GAS_fit1 <- UniGASFit(GAS_spec1, t2)
print(GAS_fit1)

Roll1 = UniGASRoll(t1, GAS_spec, ForecastLength = 50,
                   RefitEvery = 10, RefitWindow = c("moving"))
BackTest = BacktestDensity(Roll1, lower = -100, upper = 100)
BackTest$average

#---- Out-of-sample Forecasts for JSE ----
Roll_JSE <- UniGASRoll(t1,
                       GAS_spec,
                       ForecastLength = 250, 
                       RefitEvery = 10,          
                       RefitWindow = "moving"    
)

# ---- Out-of-sample Forecasts for MOEX ----
Roll_MOEX <- UniGASRoll(t2,
                        GAS_spec1,
                        ForecastLength = 250,
                        RefitEvery = 10,
                        RefitWindow = "moving"
)

# ---- Extract scale column from JSE ----
scale_JSE <- forecastMat_JSE[, "scale"]

# ---- Extract scale column from MOEX ----
forecastMat_MOEX <- getForecast(Roll_MOEX)
scale_MOEX <- forecastMat_MOEX[, "scale"]

# ---- Combine into a data frame for comparison ----
scale_comparison <- data.frame(
  Horizon = 1:length(scale_JSE),
  Scale_JSE = scale_JSE,
  Scale_MOEX = scale_MOEX
)

# ---- View first few rows ----
head(scale_comparison,10)

# ---- Plot Scale Forecasts: JSE vs MOEX ----
win.graph()
par(bg = "lightgray", mar = c(5, 5, 4, 4))
par(cex.main = 1.6, cex.axis = 1.4, cex.lab = 1.4, pch = 50)
ylim_range <- range(c(scale_comparison$Scale_JSE, scale_comparison$Scale_MOEX))
plot(
  scale_comparison$Horizon, scale_comparison$Scale_JSE, 
  type = "l", col = "blue", lwd = 2,
  xlab = "Forecast Horizon", ylab = "Scale",
  main = "Comparison of Scale Forecasts: JSE Top40 vs MOEX"
)

# Add MOEX scale on the same plot
lines(
  scale_comparison$Horizon, scale_comparison$Scale_MOEX, 
  col = "darkred", lwd = 2
)

# Add legend
legend(
  "topright", legend = c("JSE Top40", "MOEX"), 
  col = c("blue", "darkred"), lwd = 2
)
grid(col = "white", lty = "solid", lwd = 1)

