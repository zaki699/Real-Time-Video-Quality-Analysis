import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Load the dataset
data = pd.read_csv('video_quality_data.csv')

# Features for the model
features = [
    'Advanced Motion Complexity', 'DCT Complexity', 'Temporal DCT Complexity',
    'Histogram Complexity', 'Edge Detection Complexity', 'ORB Feature Complexity', 
    'Color Histogram Complexity', 'Resolution (px)', 'Frame Rate (fps)', 
    'CRF', 'average_framerate', 'min_framerate', 'max_framerate', 
    'smoothed_frame_rate_variation'
]

# Define targets for quality metrics
target_ssim = 'SSIM'
target_psnr = 'PSNR'
target_vmaf = 'VMAF'

# Determine if the encoding mode is VBR (we want to build a model for bitrate in VBR)
encoding_mode = 'VBR'  # Change this to 'CBR' if CBR mode is selected

# Adding Bitrate as a target for prediction if VBR mode is used
if encoding_mode == 'VBR':
    logger.info("Encoding mode is VBR. Including Bitrate prediction.")
    target_bitrate = 'bitrate'
else:
    logger.info("Encoding mode is CBR. Skipping Bitrate prediction.")

# Step 1: Split the data into features (X) and target labels (y)
X = data[features].values
y_ssim = data[target_ssim].values
y_psnr = data[target_psnr].values
y_vmaf = data[target_vmaf].values

# Include Bitrate if the mode is VBR
if encoding_mode == 'VBR':
    y_bitrate = data[target_bitrate].values

# Step 2: Split into training and test sets for each target
X_train, X_test, y_ssim_train, y_ssim_test = train_test_split(X, y_ssim, test_size=0.2, random_state=42)
X_train, X_test, y_psnr_train, y_psnr_test = train_test_split(X, y_psnr, test_size=0.2, random_state=42)
X_train, X_test, y_vmaf_train, y_vmaf_test = train_test_split(X, y_vmaf, test_size=0.2, random_state=42)

# If in VBR mode, also split the bitrate data
if encoding_mode == 'VBR':
    X_train, X_test, y_bitrate_train, y_bitrate_test = train_test_split(X, y_bitrate, test_size=0.2, random_state=42)

# Step 3: Define the models
# Base models for stacking
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100)),
    ('gradient_boosting', GradientBoostingRegressor()),
    ('xgboost', xgb.XGBRegressor()),
    ('lightgbm', lgb.LGBMRegressor())
]

# Step 4: Create Stacking Regressor
stacked_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Step 5: Train the models

# Neural Network for SSIM prediction
logger.info("Training model for SSIM prediction...")
stacked_model.fit(X_train, y_ssim_train)

# Neural Network for PSNR prediction
logger.info("Training model for PSNR prediction...")
stacked_model.fit(X_train, y_psnr_train)

# Neural Network for VMAF prediction
logger.info("Training model for VMAF prediction...")
stacked_model.fit(X_train, y_vmaf_train)

# Train Bitrate model only if VBR mode is enabled
if encoding_mode == 'VBR':
    logger.info("Training model for Bitrate prediction (VBR mode only)...")
    stacked_model.fit(X_train, y_bitrate_train)

# Step 6: Make predictions on the test data
# SSIM predictions
ssim_predictions = stacked_model.predict(X_test)
# PSNR predictions
psnr_predictions = stacked_model.predict(X_test)
# VMAF predictions
vmaf_predictions = stacked_model.predict(X_test)

# Bitrate predictions (only in VBR mode)
if encoding_mode == 'VBR':
    bitrate_predictions = stacked_model.predict(X_test)

# Step 7: Evaluate the predictions using mean squared error (MSE)
ssim_mse = mean_squared_error(y_ssim_test, ssim_predictions)
psnr_mse = mean_squared_error(y_psnr_test, psnr_predictions)
vmaf_mse = mean_squared_error(y_vmaf_test, vmaf_predictions)

logger.info(f"SSIM MSE: {ssim_mse}")
logger.info(f"PSNR MSE: {psnr_mse}")
logger.info(f"VMAF MSE: {vmaf_mse}")

if encoding_mode == 'VBR':
    bitrate_mse = mean_squared_error(y_bitrate_test, bitrate_predictions)
    logger.info(f"Bitrate MSE (VBR mode): {bitrate_mse}")
