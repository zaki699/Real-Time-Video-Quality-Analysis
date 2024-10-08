# Video Analysis Tool

This project provides a set of Python scripts designed for video analysis, model training, and dataset preparation using various video quality metrics, including SSIM, VMAF, and PSNR. These scripts help optimize video encoding by analyzing video complexity and quality metrics to make informed decisions about encoding settings.

## Features

- **prepare_dataset.py**: 
    - Creates or updates a CSV file that contains essential data for running video analysis on your entire catalog.
    - Supports various video analysis tasks, including frame interval customization and dynamic aspect ratio handling for multiple resolutions.
  
- **training_model.py**:
    - Builds machine learning models based on SSIM, VMAF, and PSNR metrics using the prepared CSV dataset.
    - Helps improve video compression efficiency by leveraging video quality metrics.
  
- **video_analysis.py**:
    - Analyzes video files to compute various metrics such as PSNR, SSIM, and VMAF.
    - Dynamically handles different aspect ratios, making it suitable for various video formats (e.g., 4:3, 16:9).
    - Allows frame interval configuration to balance between computational efficiency and accuracy (Default: 10 frames).
    - Finds the optimal resolution for analysis while minimizing computation time.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/video-analysis-tool.git
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Dataset
To create or update the CSV file for running analysis on your entire video catalog, use the `prepare_dataset.py` script:

```bash
python prepare_dataset.py --input_video_path /path/to/your/video --output_csv /path/to/output.csv
```

### 2. Train Model
To build the SSIM, VMAF, and PSNR models based on the dataset:

```bash
python training_model.py --input_csv /path/to/dataset.csv --output_model /path/to/sav
```

### 3. Analyze Video
To analyze a video using SSIM, PSNR, and VMAF metrics, and dynamically adjust the encoding settings:
```bash
python video_analysis.py --input_video /path/to/video --crf 23 --output_video /path/to/output/video
```

You can also configure the VMAF model path and other parameters:
```bash
python video_analysis.py --input_video /path/to/video --crf 23 --output_video /path/to/output/video --vmaf_model_path /path/to/vmaf/model
```

### Key Features
Frame Interval: Adjust the frame interval to balance computational efficiency and accuracy (Default: every 10 frames).
Dynamic Aspect Ratio Handling: The script automatically supports various aspect ratios (e.g., 4:3, 16:9), ensuring accuracy in the complexity metrics.
Optimal Resolution: The script helps you find the best resolution that provides acceptable accuracy while reducing computation time.

### Contributions
Feel free to contribute to this repository by submitting a pull request or opening an issue!
