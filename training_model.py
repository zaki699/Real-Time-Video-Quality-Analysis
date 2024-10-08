import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Step 1: Load your dataset (example)
data = pd.read_csv('video_quality_data.csv')

# Step 2: Split the data into features (X) and target labels (y)
X = data[['Scene Complexity', 'Bitrate (kbps)', 'Resolution (px)', 'Frame Rate (fps)', 'CRF']].values
y_ssim = data['SSIM'].values
y_psnr = data['PSNR'].values
y_vmaf = data['VMAF'].values

# Step 3: Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Define the function to create a Neural Network model with Dropout, BatchNorm
def create_model(input_shape, neurons=64, dropout_rate=0.5):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', input_shape=(input_shape,)))
    model.add(layers.BatchNormalization())  # Batch Normalization
    model.add(layers.Dropout(dropout_rate))  # Dropout Regularization
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())  # Batch Normalization
    model.add(layers.Dropout(dropout_rate))  # Dropout Regularization
    model.add(layers.Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Training and testing in K-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}")
    X_train, X_test = X[train_index], X[test_index]
    y_ssim_train, y_ssim_test = y_ssim[train_index], y_ssim[test_index]
    y_psnr_train, y_psnr_test = y_psnr[train_index], y_psnr[test_index]
    y_vmaf_train, y_vmaf_test = y_vmaf[train_index], y_vmaf[test_index]

    # Callbacks: EarlyStopping and ModelCheckpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_ssim_fold_{fold + 1}.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Train SSIM model
    nn_ssim_model = create_model(X_train.shape[1])
    history_ssim = nn_ssim_model.fit(X_train, y_ssim_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, checkpoint])

    # Train PSNR model
    nn_psnr_model = create_model(X_train.shape[1])
    nn_psnr_model.fit(X_train, y_psnr_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

    # Train VMAF model
    nn_vmaf_model = create_model(X_train.shape[1])
    nn_vmaf_model.fit(X_train, y_vmaf_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

    # Predictions
    ssim_predictions = nn_ssim_model.predict(X_test)
    psnr_predictions = nn_psnr_model.predict(X_test)
    vmaf_predictions = nn_vmaf_model.predict(X_test)

    # Evaluate the predictions using MSE
    ssim_mse = mean_squared_error(y_ssim_test, ssim_predictions)
    psnr_mse = mean_squared_error(y_psnr_test, psnr_predictions)
    vmaf_mse = mean_squared_error(y_vmaf_test, vmaf_predictions)

    print(f"Fold {fold + 1} Results:")
    print(f"SSIM MSE: {ssim_mse}")
    print(f"PSNR MSE: {psnr_mse}")
    print(f"VMAF MSE: {vmaf_mse}")
    print('-' * 30)

    # Plot training history for SSIM
    plt.plot(history_ssim.history['loss'], label='Training Loss')
    plt.plot(history_ssim.history['val_loss'], label='Validation Loss')
    plt.title(f'SSIM Loss Over Epochs (Fold {fold + 1})')
    plt.legend()
    plt.show()

# Optional: Hyperparameter tuning using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def create_model_tuned(neurons=64, dropout_rate=0.5):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = KerasRegressor(build_fn=create_model_tuned, epochs=100, batch_size=10)

param_grid = {
    'neurons': [32, 64, 128],
    'dropout_rate': [0.2, 0.5],
    'batch_size': [10, 20],
    'epochs': [50, 100]
}

grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3)
grid_search.fit(X_train, y_ssim_train)

print("Best parameters found:", grid_search.best_params_)
