import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('video_quality_data.csv')

# Features and target variables
X = data[['Advanced Motion Complexity', 'DCT Complexity', 'Temporal DCT Complexity',
          'Histogram Complexity', 'Edge Detection Complexity', 'ORB Feature Complexity',
          'Color Histogram Complexity', 'Bitrate (kbps)', 'Resolution (px)',
          'Frame Rate (fps)', 'CRF', 'average_framerate', 'min_framerate',
          'max_framerate', 'smoothed_frame_rate_variation']]

y_ssim = data['SSIM']
y_psnr = data['PSNR']
y_vmaf = data['VMAF']

# Split the dataset into training and testing sets
X_train, X_test, y_ssim_train, y_ssim_test = train_test_split(X, y_ssim, test_size=0.2, random_state=42)
_, _, y_psnr_train, y_psnr_test = train_test_split(X, y_psnr, test_size=0.2, random_state=42)
_, _, y_vmaf_train, y_vmaf_test = train_test_split(X, y_vmaf, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regularization Models (L1 - Lasso and L2 - Ridge)
lasso_ssim = Lasso(alpha=0.01)
ridge_ssim = Ridge(alpha=1.0)
ridge_psnr = Ridge(alpha=1.0)
ridge_vmaf = Ridge(alpha=1.0)

# Train models with L1 and L2 regularization
lasso_ssim.fit(X_train_scaled, y_ssim_train)
ridge_ssim.fit(X_train_scaled, y_ssim_train)
ridge_psnr.fit(X_train_scaled, y_psnr_train)
ridge_vmaf.fit(X_train_scaled, y_vmaf_train)

# Make predictions
ssim_lasso_predictions = lasso_ssim.predict(X_test_scaled)
ssim_ridge_predictions = ridge_ssim.predict(X_test_scaled)
psnr_ridge_predictions = ridge_psnr.predict(X_test_scaled)
vmaf_ridge_predictions = ridge_vmaf.predict(X_test_scaled)

# Evaluate models using Mean Squared Error
ssim_lasso_mse = mean_squared_error(y_ssim_test, ssim_lasso_predictions)
ssim_ridge_mse = mean_squared_error(y_ssim_test, ssim_ridge_predictions)
psnr_ridge_mse = mean_squared_error(y_psnr_test, psnr_ridge_predictions)
vmaf_ridge_mse = mean_squared_error(y_vmaf_test, vmaf_ridge_predictions)

# Print the MSE scores
print(f"SSIM Lasso MSE: {ssim_lasso_mse}")
print(f"SSIM Ridge MSE: {ssim_ridge_mse}")
print(f"PSNR Ridge MSE: {psnr_ridge_mse}")
print(f"VMAF Ridge MSE: {vmaf_ridge_mse}")

# XGBoost for feature importance evaluation
xgb_model_ssim = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model_psnr = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model_vmaf = XGBRegressor(n_estimators=100, learning_rate=0.1)

# Train the models
xgb_model_ssim.fit(X_train, y_ssim_train)
xgb_model_psnr.fit(X_train, y_psnr_train)
xgb_model_vmaf.fit(X_train, y_vmaf_train)

# Feature importance from XGBoost (SSIM model example)
feature_importance = xgb_model_ssim.feature_importances_

# Plot feature importance
def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'{model_type} Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.show()

# Call plot function for SSIM XGBoost model
plot_feature_importance(xgb_model_ssim.feature_importances_, X.columns, 'XGBoost SSIM')

# Hyperparameter tuning for XGBoost (example using cross-validation)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_ssim_train)

# Best parameters from grid search
print("Best parameters found: ", grid_search.best_params_)
best_xgb_ssim_model = grid_search.best_estimator_

# Stacking models (example combining Ridge, Lasso, and XGBoost)
from sklearn.ensemble import StackingRegressor

estimators = [
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.01)),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1))
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
stacking_model.fit(X_train_scaled, y_ssim_train)

# Make predictions using the stacking model
stacking_predictions = stacking_model.predict(X_test_scaled)
stacking_mse = mean_squared_error(y_ssim_test, stacking_predictions)

print(f"Stacking Model MSE: {stacking_mse}")

# Advanced Feature Engineering (Example: Polynomial Features)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y_ssim, test_size=0.2, random_state=42)
ridge_model_poly = Ridge(alpha=1.0)
ridge_model_poly.fit(X_train_poly, y_train)

# Predictions with polynomial features
poly_predictions = ridge_model_poly.predict(X_test_poly)
poly_mse = mean_squared_error(y_test, poly_predictions)

print(f"Polynomial Feature Ridge MSE: {poly_mse}")

# Evaluate Model with Cross-Validation
cv_scores = cross_val_score(ridge_ssim, X_train_scaled, y_ssim_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {-cv_scores.mean()}")
