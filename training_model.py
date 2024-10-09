# Step 1: Load your dataset (example)
import pandas as pd
from pydantic import create_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('video_quality_data.csv')

# Step 2: Split the data into features (X) and target labels (y)
X = data[['Scene Complexity', 'Bitrate (kbps)', 'Resolution (px)', 'Frame Rate (fps)', 'CRF']].values
y_ssim = data['SSIM'].values
y_psnr = data['PSNR'].values
y_vmaf = data['VMAF'].values

# Step 3: Split into training and test sets for each target
X_train, X_test, y_ssim_train, y_ssim_test = train_test_split(X, y_ssim, test_size=0.2, random_state=42)
X_train, X_test, y_psnr_train, y_psnr_test = train_test_split(X, y_psnr, test_size=0.2, random_state=42)
X_train, X_test, y_vmaf_train, y_vmaf_test = train_test_split(X, y_vmaf, test_size=0.2, random_state=42)

# Step 4: Call the Neural Network Model Creation function and train the models

# Neural Network for SSIM prediction
nn_ssim_model = create_model(X_train.shape[1])
nn_ssim_model.fit(X_train, y_ssim_train, epochs=100, validation_split=0.2)

# Neural Network for PSNR prediction
nn_psnr_model = create_model(X_train.shape[1])
nn_psnr_model.fit(X_train, y_psnr_train, epochs=100, validation_split=0.2)

# Neural Network for VMAF prediction
nn_vmaf_model = create_model(X_train.shape[1])
nn_vmaf_model.fit(X_train, y_vmaf_train, epochs=100, validation_split=0.2)

# Step 5: Make predictions on the test data

# SSIM predictions
ssim_predictions = nn_ssim_model.predict(X_test)
# PSNR predictions
psnr_predictions = nn_psnr_model.predict(X_test)
# VMAF predictions
vmaf_predictions = nn_vmaf_model.predict(X_test)

# Step 6: Evaluate the predictions (e.g., print results or calculate error)
print("SSIM Predictions:", ssim_predictions)
print("PSNR Predictions:", psnr_predictions)
print("VMAF Predictions:", vmaf_predictions)

# Optional: Evaluate the models by calculating mean squared error (MSE)
ssim_mse = mean_squared_error(y_ssim_test, ssim_predictions)
psnr_mse = mean_squared_error(y_psnr_test, psnr_predictions)
vmaf_mse = mean_squared_error(y_vmaf_test, vmaf_predictions)

print(f"SSIM MSE: {ssim_mse}")
print(f"PSNR MSE: {psnr_mse}")
print(f"VMAF MSE: {vmaf_mse}")
