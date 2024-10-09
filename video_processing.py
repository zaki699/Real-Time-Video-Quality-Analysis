import argparse
import json
import multiprocessing
import re
import tempfile
import shutil
import os
import threading
import uuid
import pandas as pd
import subprocess
from logging.handlers import QueueHandler, QueueListener
import queue
import logging
from concurrent.futures import ProcessPoolExecutor as Executor
import functools

# Import the calculate_average_scene_complexity function
from complexity_metrics import calculate_average_scene_complexity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a thread-safe queue for logging
log_queue = queue.Queue()

# Define a log handler that is thread-safe by using a QueueHandler
queue_handler = QueueHandler(log_queue)
log_file_handler = logging.FileHandler('video_processing.log')

# Add the log handlers to the logger
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)

# Create a listener that will listen to the queue and write logs to the file
listener = QueueListener(log_queue, log_file_handler)
listener.start()

# To ensure thread safety when writing to the CSV, we use a file lock:
log_lock = threading.Lock()

# Thread-safe CSV update
def thread_safe_update_csv(metrics, csv_file='video_quality_data.csv'):
    """
    Thread-safe function to update a CSV file with new metrics.

    Parameters:
        metrics (dict): Dictionary containing the metrics to write.
        csv_file (str): Path to the CSV file.

    Returns:
        None
    """
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    # Create DataFrame from metrics
    df = pd.DataFrame([metrics])

    # Thread-safe CSV writing
    with log_lock:
        try:
            with open(csv_file, 'a', newline='') as f:
                df.to_csv(f, index=False, header=not file_exists)
        except IOError as e:
            logger.error("Failed to write to CSV file: %s", e)
            raise

# Load and validate configuration from JSON file
def load_config(config_file):
    """
    Loads and validates the configuration from the JSON file.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        validate_config(config)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file {config_file}.")
        raise

def validate_config(config):
    """
    Validates the required config values and ensures that they're within reasonable ranges.
    """
    if not (1 <= config.get('crf', 23) <= 51):
        raise ValueError("CRF value must be between 1 and 51.")
    if config.get('resize_width', 0) <= 0 or config.get('resize_height', 0) <= 0:
        raise ValueError("Resize dimensions must be positive integers.")
    if config.get('frame_interval', 10) <= 0:
        raise ValueError("Frame interval must be a positive integer.")
    if not isinstance(config.get('num_workers', multiprocessing.cpu_count() // 2), int):
        raise ValueError("num_workers must be an integer.")

# Get video info using ffprobe
def get_video_info(video_path):
    """
    Uses ffprobe to retrieve video properties like bitrate, resolution, and frame rate.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        tuple: (bitrate in kbps, resolution as string, frame_rate, width, height)
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-print_format', 'json',
        '-show_entries',
        'stream=width,height,avg_frame_rate,bit_rate',
        video_path
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error("ffprobe failed to retrieve video information.")
            raise RuntimeError("ffprobe failed.")

        data = json.loads(stdout)
        stream = data['streams'][0]
        bitrate = int(stream.get('bit_rate', 0)) // 1000  # Convert to kbps
        width = stream.get('width', 0)
        height = stream.get('height', 0)
        resolution = f"{width}x{height}"
        avg_frame_rate = stream.get('avg_frame_rate', '0/1')
        frame_rate = eval(avg_frame_rate) if avg_frame_rate != '0/0' else 0

        return bitrate, resolution, frame_rate, width, height

    except Exception as e:
        logger.error(f"Error retrieving video information: {e}")
        raise

# Extract metrics from logs
def extract_metrics_from_logs(psnr_log, ssim_log, vmaf_log, video_file, crf, bitrate, resolution, frame_rate):
    """
    Extract PSNR, SSIM, and VMAF metrics from their respective log files.
    """
    metrics = {
        'Bitrate (kbps)': bitrate,
        'Resolution (px)': resolution,
        'Frame Rate (fps)': frame_rate,
        'CRF': crf
    }

    try:
        if os.path.isfile(psnr_log):
            with open(psnr_log) as f:
                content = f.read()
                match = re.search(r'psnr_avg:(\s*\d+\.\d+)', content)
                if match:
                    metrics['PSNR'] = float(match.group(1))
        if os.path.isfile(ssim_log):
            with open(ssim_log) as f:
                content = f.read()
                match = re.search(r'All:(\s*\d+\.\d+)', content)
                if match:
                    metrics['SSIM'] = float(match.group(1))
        if os.path.isfile(vmaf_log):
            with open(vmaf_log) as f:
                vmaf_data = json.load(f)
                if 'pooled_metrics' in vmaf_data and 'vmaf' in vmaf_data['pooled_metrics']:
                    metrics['VMAF'] = vmaf_data['pooled_metrics']['vmaf']['mean']
    except Exception as e:
        logger.warning(f"Error extracting metrics from logs: {e}")

    return metrics

# Process video and extract metrics
def process_video_and_extract_metrics(input_video, config):
    """
    Process the input video, calculate scene complexity, and extract quality metrics.
    """
    crf = config.get("crf", 23)
    vmaf_model_path = config.get("vmaf_model_path", None)
    resize_width = config.get("resize_width", 64)
    resize_height = config.get("resize_height", 64)
    frame_interval = config.get("frame_interval", 10)

    unique_id = uuid.uuid4().hex
    psnr_log = os.path.join(tempfile.gettempdir(), f'psnr_{unique_id}.log')
    ssim_log = os.path.join(tempfile.gettempdir(), f'ssim_{unique_id}.log')
    vmaf_log = os.path.join(tempfile.gettempdir(), f'vmaf_{unique_id}.json')

    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"The input video file {input_video} does not exist.")

    temp_dir = tempfile.mkdtemp()
    try:
        encoded_video = os.path.join(temp_dir, 'encoded_video.mp4')
        encode_cmd = [
            'ffmpeg',
            '-i', input_video,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-y',  # Overwrite output file if it exists
            encoded_video
        ]
        try:
            subprocess.run(encode_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            raise

        run_ffmpeg_metrics(input_video, encoded_video, psnr_log, ssim_log, vmaf_log, vmaf_model_path)

        bitrate, resolution, frame_rate, _, _ = get_video_info(input_video)

        metrics = extract_metrics_from_logs(
            psnr_log=psnr_log,
            ssim_log=ssim_log,
            vmaf_log=vmaf_log,
            video_file=input_video,
            crf=crf,
            bitrate=bitrate,
            resolution=resolution,
            frame_rate=frame_rate
        )

        logger.info("Metrics extracted: %s", metrics)

        # Call calculate_average_scene_complexity after encoding
        logger.info("Calculating scene complexity after encoding...")
        (advanced_motion_complexity,
         dct_complexity,
         temporal_dct_complexity,
         histogram_complexity,
         edge_detection_complexity,
         orb_feature_complexity,
         color_histogram_complexity) = calculate_average_scene_complexity(
             encoded_video,
             resize_width,
             resize_height,
             frame_interval=frame_interval,
         )
        

        metrics.update({
            'Advanced Motion Complexity': advanced_motion_complexity,
            'DCT Complexity': dct_complexity,
            'Temporal DCT Complexity': temporal_dct_complexity,
            'Histogram Complexity': histogram_complexity,
            'Edge Detection Complexity': edge_detection_complexity,
            'ORB Feature Complexity': orb_feature_complexity,
            'Color Histogram Complexity': color_histogram_complexity
        })

        thread_safe_update_csv(metrics, csv_file='video_quality_data.csv')

    finally:
        for log_file in [psnr_log, ssim_log, vmaf_log]:
            if os.path.exists(log_file):
                os.remove(log_file)
        shutil.rmtree(temp_dir)

# Run FFmpeg to compute PSNR, SSIM, and VMAF
def run_ffmpeg_metrics(reference_video, distorted_video, psnr_log, ssim_log, vmaf_log, vmaf_model_path=None):
    """
    Run FFmpeg to compute PSNR, SSIM, and VMAF between the reference and distorted videos.
    """
    filters = [
        f"[0:v][1:v]psnr=stats_file={psnr_log}",
        f"[0:v][1:v]ssim=stats_file={ssim_log}",
    ]

    if vmaf_model_path and os.path.isfile(vmaf_model_path):
        filters.append(f"[0:v][1:v]libvmaf=model_path={vmaf_model_path}:log_path={vmaf_log}:log_fmt=json")
    else:
        filters.append(f"[0:v][1:v]libvmaf=log_path={vmaf_log}:log_fmt=json")

    cmd = [
        'ffmpeg',
        '-i', distorted_video,
        '-i', reference_video,
        '-filter_complex', ';'.join(filters),
        '-f', 'null',
        '-'
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg metrics calculation failed: {e}")
        raise

# Main function for parsing arguments and running the processing
def main():
    parser = argparse.ArgumentParser(description="Process a video, extract metrics, and update CSV.")
    parser.add_argument('config_file', type=str, help="Path to the configuration JSON file.")
    parser.add_argument('input_video', type=str, help="Path to the input video file.")
    
    args = parser.parse_args()

    config = load_config(args.config_file)

    try:
        process_video_and_extract_metrics(
            input_video=args.input_video,
            config=config
        )
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise

if __name__ == "__main__":
    print("Main function is being executed.")
    main()
    listener.stop()
