import argparse
import json
import re
import tempfile
import shutil
import os
import uuid
import pandas as pd
import os
import threading
import subprocess
from logging.handlers import QueueHandler, QueueListener
import queue
import threading
import logging
from concurrent.futures import ProcessPoolExecutor as Executor

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

# Load configuration from JSON file
def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file {config_file}.")
        raise

def get_video_info(video_path):
    """
    Uses ffprobe to retrieve video properties like bitrate, resolution, and frame rate.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        tuple: (bitrate in kbps, resolution as string, frame_rate, width, height)
    """
    # Use ffprobe to retrieve video properties
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

        # Parse bitrate, resolution, frame rate, width, and height from stdout
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

def extract_metrics_from_logs(psnr_log, ssim_log, vmaf_log, video_file, crf, bitrate, resolution, frame_rate, resize_width, resize_height, frame_interval=10):
    # Initialize metrics
    psnr = None
    ssim = None
    vmaf = None

    # Extract PSNR value from the log file
    if os.path.isfile(psnr_log):
        with open(psnr_log) as f:
            content = f.read()
            match = re.search(r'psnr_avg:(\s*\d+\.\d+)', content)
            if match:
                psnr = float(match.group(1))
    else:
        logger.warning("PSNR log file not found.")

    # Extract SSIM value from the log file
    if os.path.isfile(ssim_log):
        with open(ssim_log) as f:
            content = f.read()
            match = re.search(r'All:(\s*\d+\.\d+)', content)
            if match:
                ssim = float(match.group(1))
    else:
        logger.warning("SSIM log file not found.")

    # Extract VMAF value from the log file
    if os.path.isfile(vmaf_log):
        with open(vmaf_log) as f:
            vmaf_data = json.load(f)
            if 'pooled_metrics' in vmaf_data and 'vmaf' in vmaf_data['pooled_metrics']:
                vmaf = vmaf_data['pooled_metrics']['vmaf']['mean']
    else:
        logger.warning("VMAF log file not found.")

    # Calculate scene complexity
    logger.info("Calculating average scene complexity...")

    return {
        'Bitrate (kbps)': bitrate,
        'Resolution (px)': resolution,
        'Frame Rate (fps)': frame_rate,
        'CRF': crf,
        'SSIM': ssim,
        'PSNR': psnr,
        'VMAF': vmaf
    }
    
def process_video_and_extract_metrics(input_video, config):
    crf = config.get("crf", 23)
    vmaf_model_path = config.get("vmaf_model_path", None)
    resize_width = config.get("resize_width", 64)
    resize_height = config.get("resize_height", 64)
    frame_interval = config.get("frame_interval", 10)
    smoothing_factor = config.get("smoothing_factor", 0.8)
    min_frames_for_parallel = config.get("min_frames_for_parallel", 50)

    # Generate unique filenames for log files to avoid conflicts
    unique_id = uuid.uuid4().hex
    psnr_log = os.path.join(tempfile.gettempdir(), f'psnr_{unique_id}.log')
    ssim_log = os.path.join(tempfile.gettempdir(), f'ssim_{unique_id}.log')
    vmaf_log = os.path.join(tempfile.gettempdir(), f'vmaf_{unique_id}.json')
    
    # Validate input video
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"The input video file {input_video} does not exist.")

    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    try:
        # Encode the input video with the specified CRF value to a temporary file
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
        subprocess.run(encode_cmd, check=True)

        # Run FFmpeg to compute metrics between the original and encoded videos
        run_ffmpeg_metrics(input_video, encoded_video, psnr_log, ssim_log, vmaf_log,vmaf_model_path)

        # Extract bitrate, resolution, and frame rate from the input video
        bitrate, resolution, frame_rate, _, _ = get_video_info(input_video)

        # Extract metrics from the generated logs (PSNR, SSIM, VMAF)
        metrics = extract_metrics_from_logs(
            psnr_log=psnr_log,
            ssim_log=ssim_log,
            vmaf_log=vmaf_log,
            video_file=input_video,
            crf=crf,
            bitrate=bitrate,
            resolution=resolution,
            frame_rate=frame_rate,
            resize_width=resize_width,
            resize_height=resize_height,
            frame_interval=frame_interval
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
             smoothing_factor=smoothing_factor,
             min_frames_for_parallel=min_frames_for_parallel
         )

        # Add the scene complexity metrics to the existing metrics dictionary
        metrics.update({
            'Advanced Motion Complexity': advanced_motion_complexity,
            'DCT Complexity': dct_complexity,
            'Temporal DCT Complexity': temporal_dct_complexity,
            'Histogram Complexity': histogram_complexity,
            'Edge Detection Complexity': edge_detection_complexity,
            'ORB Feature Complexity': orb_feature_complexity,
            'Color Histogram Complexity': color_histogram_complexity
        })

        # Update the CSV file with the new metrics
        thread_safe_update_csv(metrics, csv_file='video_quality_data.csv')

    finally:
        # Clean up temporary log files
        if os.path.exists(psnr_log):
            os.remove(psnr_log)
        if os.path.exists(ssim_log):
            os.remove(ssim_log)
        if os.path.exists(vmaf_log):
            os.remove(vmaf_log)
        # Clean up temporary files
        shutil.rmtree(temp_dir)

# Function to run FFmpeg to compute PSNR, SSIM, and VMAF
def run_ffmpeg_metrics(reference_video, distorted_video, psnr_log, ssim_log, vmaf_log,vmaf_model_path=None):
    # Construct FFmpeg command
    ffmpeg_path = 'ffmpeg'


    # Enclose paths with spaces in single quotes
    def quote_path(path):
        return f"'{path}'" if ' ' in path else path

    psnr_log_escaped = quote_path({psnr_log})
    ssim_log_escaped = quote_path({ssim_log})
    vmaf_log_escaped = quote_path({vmaf_log})

    filters = []
    filters.append(f"[0:v][1:v]psnr=stats_file={psnr_log}")
    filters.append(f"[0:v][1:v]ssim=stats_file={ssim_log}")

    # Build the libvmaf filter options
    libvmaf_options = []
    if vmaf_model_path:
        # Validate VMAF model path
        if not os.path.isfile(vmaf_model_path):
            raise FileNotFoundError(f"VMAF model not found at {vmaf_model_path}")
        vmaf_model_path_escaped = quote_path(vmaf_model_path)
        libvmaf_options.append(f"model_path={vmaf_model_path_escaped}")
    else:
        logger.info("VMAF model path not provided; using FFmpeg's default VMAF model.")

    libvmaf_options.append(f"log_path={vmaf_log}")
    libvmaf_options.append("log_fmt=json")

    # Combine libvmaf options
    libvmaf_options_str = ':'.join(libvmaf_options)
    filters.append(f"[0:v][1:v]libvmaf={libvmaf_options_str}")

    filter_complex = ';'.join(filters)

    cmd = [
        ffmpeg_path,
        '-i', distorted_video,
        '-i', reference_video,
        '-filter_complex', filter_complex,
        '-f', 'null',
        '-'
    ]

    logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        logger.error("FFmpeg process failed.")
        logger.error("FFmpeg stderr: %s", stderr.decode())
        raise RuntimeError(f"FFmpeg process failed with return code {process.returncode}")



# Main function for parsing arguments and running the processing
def main():
    parser = argparse.ArgumentParser(description="Process a video, extract metrics, and update CSV.")
    parser.add_argument('config_file', type=str, help="Path to the configuration JSON file.")
    parser.add_argument('input_video', type=str, help="Path to the input video file.")
    
    args = parser.parse_args()

    # Load configuration from JSON file
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
