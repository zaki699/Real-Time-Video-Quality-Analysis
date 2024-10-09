# Video Complexity and Quality Metric Analysis

This project analyzes video complexity using various metrics such as advanced motion complexity, DCT complexity, temporal DCT complexity, histogram complexity, edge detection, ORB feature complexity, and color histogram complexity. Additionally, it computes video quality metrics like PSNR, SSIM, and VMAF using FFmpeg.

## Features

- **Frame-based Scene Complexity Analysis**: Calculates complexity using motion estimation, DCT, histogram, edge detection, and ORB features.
- **Quality Metrics Calculation**: Computes PSNR, SSIM, and VMAF between original and compressed videos.
- **Parallel Processing**: Processes video frames in parallel to speed up computation using batch processing.
- **Exponential Smoothing**: Smooths metrics over time to reduce noise in complexity measurements.
- **Configurable Parameters**: Easily configure CRF, frame interval, and video resizing in a JSON file.

## Requirements

- Python 3.6+
- FFmpeg (installed and available in the system's PATH)
- Python packages: Install them by running:

  ```bash
  pip install -r requirements.txt
  ```

  The requirements.txt should include:

  - OpenCV
  - NumPy
  - Pandas
  - tqdm
  - logging
  - json
  - multiprocessing

## Configuration

You can configure the parameters in config.json. The default parameters are as follows:

```Json
{
    "crf": 23,
    "vmaf_model_path": null,
    "resize_width": 64,
    "resize_height": 64,
    "frame_interval": 10
}
```

- crf: CRF (Constant Rate Factor) for video encoding.
- vmaf_model_path: Path to the VMAF model file. If not specified, FFmpeg’s default model is used.
- resize_width: Width to which frames are resized for processing.
- resize_height: Height to which frames are resized for processing.
- frame_interval: Interval of frames to process for complexity analysis.


## Usage
To run the project, use the following command:

```bash
python video_processing.py config.json input_video.mp4
```

## Example
```bash
python video_processing.py config.json sample_video.mp4
```

This will process the sample_video.mp4 file and calculate the complexity and quality metrics, saving the results to video_quality_data.csv.

### Functionality Overview

## Key Functions

- calculate_average_scene_complexity: Calculates complexity metrics like motion, DCT, temporal DCT, and edge detection.
- process_in_batches: Handles the parallel processing of frames in batches.
- run_ffmpeg_metrics: Executes FFmpeg commands to calculate PSNR, SSIM, and VMAF.
- thread_safe_update_csv: Updates CSV files in a thread-safe manner to store results.

## Output

The results of the analysis are saved in a CSV file (video_quality_data.csv). The following metrics are included:

	- Advanced Motion Complexity
	- DCT Complexity
	- Temporal DCT Complexity
	- ORB Feature Complexity
	- Histogram Complexity
	- Edge Detection Complexity
	- Color Histogram Complexity
	- PSNR, SSIM, and VMAF (Quality Metrics)
	- Bitrate, Frame Rate, and Resolution

## Logging

Logs are stored in video_processing.log, providing details of the processing steps, warnings, and errors.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


# Video Encoding Quality Prediction Models

This project aims to predict video quality metrics such as **SSIM**, **PSNR** and **VMAF**  based on video complexity features, bitrate, resolution, frame rate, and other encoding settings. In **VBR** (Variable Bitrate) mode, the project also predicts the **bitrate** of the video.

## Features

The dataset includes the following metrics and features:

- **Advanced Motion Complexity**
- **DCT Complexity**
- **Temporal DCT Complexity**
- **Histogram Complexity**
- **Edge Detection Complexity**
- **ORB Feature Complexity**
- **Color Histogram Complexity**
- **Bitrate (kbps)** (for VBR encoding mode only)
- **Resolution (px)**
- **Frame Rate (fps)**
- **CRF (Constant Rate Factor)**
- **SSIM** (Structural Similarity Index)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **VMAF** (Video Multi-Method Assessment Fusion)
- **average_framerate**
- **min_framerate**
- **max_framerate**
- **smoothed_frame_rate_variation**

## Models

The project uses the following ensemble machine learning models to predict video quality metrics:

1. **Random Forest Regressor**
2. **Gradient Boosting Regressor**
3. **XGBoost Regressor**
4. **LightGBM Regressor**
5. **Linear Regression** as the final meta-learner in the stacking model.

### Bitrate Prediction

In **VBR** (Variable Bitrate) mode, the project also predicts **bitrate** based on the same set of features. **Bitrate prediction is skipped in CBR (Constant Bitrate) mode.**

## Requirements

The following Python libraries are required for this project:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `logging`
- `tqdm`

You can install the dependencies using the following command:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm tqdm
```

## Usage

1. Load the dataset

Ensure that your dataset is saved as video_quality_data.csv. It should include the features listed above.

2. Modify the Encoding Mode

You can set the encoding mode to either 'VBR' (Variable Bitrate) or 'CBR' (Constant Bitrate). This will control whether the bitrate prediction model is built or skipped.

```Python
encoding_mode = 'VBR'  # or 'CBR'
```

3. Running the Code

Once the dataset is ready, and the encoding mode is set, you can run the script to train the models and make predictions for SSIM, PSNR, VMAF, and bitrate (if VBR).

4. Training and Predictions

The models are trained using the features in the dataset. The predictions for the test data are made for SSIM, PSNR, VMAF, and bitrate (if VBR). The Mean Squared Error (MSE) for each metric is printed for evaluation.

5. Evaluating Model Performance

The following metrics are used for evaluation:

	- Mean Squared Error (MSE): The lower the MSE, the better the model.

Example Output

After running the script, you should see the following output for MSE:

```bash
SSIM MSE: 0.0015
PSNR MSE: 0.0032
VMAF MSE: 0.0021
Bitrate MSE (VBR mode): 100.5  # Only shown in VBR mode
```

Future Improvements

	- Advanced Feature Engineering: Further improvements can be made by exploring feature interaction and generating additional derived features.
	- Hyperparameter Tuning: Use Grid Search or Random Search to fine-tune the model hyperparameters.
	- Cross-Validation: Implement k-fold cross-validation to ensure the model is not overfitting to the training data.
	- Model Interpretability: Consider using techniques like SHAP or LIME for better interpretability of the models.
