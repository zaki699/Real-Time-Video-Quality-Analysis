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
