import argparse
import re
import subprocess
import cv2
import numpy as np
import pandas as pd
import os
import sys
import logging
import time
import math
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Smoothing function for motion complexity and other metrics
def smooth_data(data, smoothing_factor=0.8):
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("Data must be a list or NumPy array.")
    if len(data) == 0:
        return np.array([])
    smoothed_data = np.zeros(len(data))
    smoothed_data[0] = data[0]  # Initialize with the first value

    for i in range(1, len(data)):
        smoothed_data[i] = (smoothing_factor * data[i] +
                            (1 - smoothing_factor) * smoothed_data[i - 1])
    return smoothed_data

# Process frames in parallel for efficiency
def process_frame_complexity(frame, prev_frame):
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow between consecutive frames
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_motion = np.mean(mag)
    return avg_motion

def calculate_advanced_motion_complexity(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_motion = []
    frames = []
    frame_count = 0

    try:
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("Error: Unable to read video.")
        frame_count = 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            frames.append((frame, prev_frame))
            prev_frame = frame
    finally:
        cap.release()

    # Use ThreadPoolExecutor to parallelize the frame processing
    with ThreadPoolExecutor() as executor:
        complexities = list(executor.map(lambda f: process_frame_complexity(f[0], f[1]), frames))

    smoothed_motion = smooth_data(complexities)
    return np.mean(smoothed_motion) if smoothed_motion else 0.0

# DCT complexity calculation (can also be parallelized)
def calculate_dct_scene_complexity(video_path, resize_width, resize_height, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_dct_energy = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))

            dct_frame = cv2.dct(np.float32(gray_frame))
            energy = np.sum(dct_frame ** 2)
            total_dct_energy.append(energy)
    finally:
        cap.release()

    smoothed_dct = smooth_data(total_dct_energy)
    return np.mean(smoothed_dct) if len(smoothed_dct) > 0 else 0.0

# Temporal DCT complexity
def calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_temporal_dct_energy = []
    frame_count = 0
    prev_frame_dct = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))
            curr_frame_dct = cv2.dct(np.float32(gray_frame))

            if prev_frame_dct is not None:
                temporal_energy = np.sum((curr_frame_dct - prev_frame_dct) ** 2)
                total_temporal_dct_energy.append(temporal_energy)

            prev_frame_dct = curr_frame_dct
    finally:
        cap.release()

    smoothed_temporal_dct = smooth_data(total_temporal_dct_energy)
    return np.mean(smoothed_temporal_dct) if len(smoothed_temporal_dct) > 0 else 0.0

# Histogram complexity
def calculate_histogram_complexity(video_path, resize_width, resize_height, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_entropy = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            total_entropy.append(entropy)
    finally:
        cap.release()

    smoothed_entropy = smooth_data(total_entropy)
    return np.mean(smoothed_entropy) if len(smoothed_entropy) > 0 else 0.0

# Edge Detection complexity
def calculate_edge_detection_complexity(video_path, resize_width, resize_height, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_edges = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_frame, 100, 200)
            edge_count = np.sum(edges > 0)
            total_edges.append(edge_count)
    finally:
        cap.release()

    smoothed_edges = smooth_data(total_edges)
    return np.mean(smoothed_edges) if len(smoothed_edges) > 0 else 0.0

# Color Complexity (new metric)
def calculate_color_complexity(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_variance = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            variance = np.var(frame, axis=(0, 1))  # Variance of color channels (R, G, B)
            total_variance.append(np.mean(variance))
    finally:
        cap.release()

    smoothed_variance = smooth_data(total_variance)
    return np.mean(smoothed_variance) if len(smoothed_variance) > 0 else 0.0

# Normalization function
def min_max_normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.0
    return (value - min_val) / (max_val - min_val)

# Final average scene complexity with normalization and weighted average
def calculate_average_scene_complexity(video_path, resize_width, resize_height, frame_interval=10):
    logger.info("Calculating advanced motion complexity...")
    advanced_motion_complexity = calculate_advanced_motion_complexity(video_path, frame_interval)
    logger.info("Calculating DCT scene complexity...")
    dct_complexity = calculate_dct_scene_complexity(video_path, resize_width, resize_height, frame_interval)
    logger.info("Calculating temporal DCT complexity...")
    temporal_dct_complexity = calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval)
    logger.info("Calculating histogram complexity...")
    histogram_complexity = calculate_histogram_complexity(video_path, resize_width, resize_height, frame_interval)
    logger.info("Calculating edge detection complexity...")
    edge_detection_complexity = calculate_edge_detection_complexity(video_path, resize_width, resize_height, frame_interval)
    logger.info("Calculating color complexity...")
    color_complexity = calculate_color_complexity(video_path, frame_interval)

    # Define min and max values for normalization
    metric_min_values = {
        'Advanced Motion Complexity': 0.0,
        'DCT Complexity': 1e6,
        'Temporal DCT Complexity': 0.0,
        'Histogram Complexity': 0.0,
        'Edge Detection Complexity': 0.0,
        'Color Complexity': 0.0
    }

    metric_max_values = {
        'Advanced Motion Complexity': 10.0,
        'DCT Complexity': 5e7,
        'Temporal DCT Complexity': 1e7,
        'Histogram Complexity': 8.0,
        'Edge Detection Complexity': resize_width * resize_height,
        'Color Complexity': 30000.0
    }

    # Normalize metrics
    normalized_metrics = {}
    normalized_metrics['Advanced Motion Complexity'] = min_max_normalize(
        advanced_motion_complexity, metric_min_values['Advanced Motion Complexity'], metric_max_values['Advanced Motion Complexity'])
    normalized_metrics['DCT Complexity'] = min_max_normalize(
        dct_complexity, metric_min_values['DCT Complexity'], metric_max_values['DCT Complexity'])
    normalized_metrics['Temporal DCT Complexity'] = min_max_normalize(
        temporal_dct_complexity, metric_min_values['Temporal DCT Complexity'], metric_max_values['Temporal DCT Complexity'])
    normalized_metrics['Histogram Complexity'] = min_max_normalize(
        histogram_complexity, metric_min_values['Histogram Complexity'], metric_max_values['Histogram Complexity'])
    normalized_metrics['Edge Detection Complexity'] = min_max_normalize(
        edge_detection_complexity, metric_min_values['Edge Detection Complexity'], metric_max_values['Edge Detection Complexity'])
    normalized_metrics['Color Complexity'] = min_max_normalize(
        color_complexity, metric_min_values['Color Complexity'], metric_max_values['Color Complexity'])

    # Define weights for each metric
    weights = {
        'Advanced Motion Complexity': 0.25,
        'DCT Complexity': 0.20,
        'Temporal DCT Complexity': 0.20,
        'Histogram Complexity': 0.15,
        'Edge Detection Complexity': 0.10,
        'Color Complexity': 0.10
    }

    # Calculate weighted average
    total_weight = sum(weights.values())
    weighted_sum = sum(normalized_metrics[metric] * weights[metric] for metric in normalized_metrics)
    overall_complexity = weighted_sum / total_weight

    logger.info(f"Normalized Metrics: {normalized_metrics}")
    logger.info(f"Overall Scene Complexity (Weighted Average): {overall_complexity:.4f}")

    return overall_complexity

# Function to run FFmpeg to compute PSNR, SSIM, and VMAF
def run_ffmpeg_metrics(reference_video, distorted_video, vmaf_model_path=None):
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct FFmpeg command
        ffmpeg_path = 'ffmpeg'
        psnr_log = os.path.join(temp_dir, 'psnr.log')
        ssim_log = os.path.join(temp_dir, 'ssim.log')
        vmaf_log = os.path.join(temp_dir, 'vmaf.json')

        # Enclose paths with spaces in single quotes
        def quote_path(path):
            return f"'{path}'" if ' ' in path else path

        psnr_log_escaped = quote_path(psnr_log)
        ssim_log_escaped = quote_path(ssim_log)
        vmaf_log_escaped = quote_path(vmaf_log)

        filters = []
        filters.append(f"[0:v][1:v]psnr=stats_file={psnr_log_escaped}")
        filters.append(f"[0:v][1:v]ssim=stats_file={ssim_log_escaped}")

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

        libvmaf_options.append(f"log_path={vmaf_log_escaped}")
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

        # For debugging, print the command
        logger.debug("FFmpeg Command: %s", ' '.join(cmd))

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Print FFmpeg output for debugging
        logger.debug("FFmpeg Output: %s", stdout.decode())
        logger.debug("FFmpeg Error Output: %s", stderr.decode())

        if process.returncode != 0:
            logger.error("FFmpeg process failed with return code %s", process.returncode)
            logger.error("FFmpeg stderr: %s", stderr.decode())
            raise RuntimeError(f"FFmpeg process failed with return code {process.returncode}")

        # Return the paths of the temporary logs
        return psnr_log, ssim_log, vmaf_log

# Metrics extraction function to read logs and calculate metrics
def extract_metrics_from_logs(psnr_log, ssim_log, vmaf_log, video_file, crf, bitrate, resolution, frame_rate, resize_width, resize_height, frame_interval=10):
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
        import json
        with open(vmaf_log) as f:
            vmaf_data = json.load(f)
            if 'pooled_metrics' in vmaf_data and 'vmaf' in vmaf_data['pooled_metrics']:
                vmaf = vmaf_data['pooled_metrics']['vmaf']['mean']
    else:
        logger.warning("VMAF log file not found.")

    # Calculate scene complexity
    logger.info("Calculating average scene complexity...")
    scene_complexity = calculate_average_scene_complexity(video_file, resize_width, resize_height, frame_interval)

    return {
        'Scene Complexity': scene_complexity,
        'Bitrate (kbps)': bitrate,
        'Resolution (px)': resolution,
        'Frame Rate (fps)': frame_rate,
        'CRF': crf,
        'SSIM': ssim,
        'PSNR': psnr,
        'VMAF': vmaf
    }

def update_csv(metrics, csv_file='video_quality_data.csv'):
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    # Create DataFrame from metrics
    df = pd.DataFrame([metrics])

    # Handle concurrent writes
    try:
        with open(csv_file, 'a', newline='') as f:
            df.to_csv(f, index=False, header=not file_exists)
    except IOError as e:
        logger.error("Failed to write to CSV file: %s", e)
        raise

def get_video_info(video_path):
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

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        logger.error("ffprobe failed to retrieve video information.")
        raise RuntimeError("ffprobe failed.")

    # Parse bitrate, resolution, frame rate, width, and height from stdout
    import json
    data = json.loads(stdout)
    stream = data['streams'][0]
    bitrate = int(stream.get('bit_rate', 0)) // 1000  # Convert to kbps
    width = stream.get('width', 0)
    height = stream.get('height', 0)
    resolution = f"{width}x{height}"
    avg_frame_rate = stream.get('avg_frame_rate', '0/1')
    frame_rate = eval(avg_frame_rate) if avg_frame_rate != '0/0' else 0

    return bitrate, resolution, frame_rate, width, height

def calculate_aspect_ratio(width, height):
    gcd = math.gcd(width, height)
    width_ratio = width // gcd
    height_ratio = height // gcd
    return width_ratio, height_ratio

def generate_resolutions(aspect_ratio, base_heights):
    width_ratio, height_ratio = aspect_ratio
    resolutions = []
    for base_height in base_heights:
        base_width = int((base_height * width_ratio) / height_ratio)
        resolutions.append((base_width, base_height))
    return resolutions

# Main function to process video and extract metrics
def process_video_and_extract_metrics(input_video, crf, output_video, vmaf_model_path, resize_width, resize_height, frame_interval=10):
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
        psnr_log, ssim_log, vmaf_log = run_ffmpeg_metrics(input_video, encoded_video, vmaf_model_path)

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

        # Update the CSV file with the new metrics
        update_csv(metrics, csv_file='video_quality_data.csv')
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)

def parse_arguments():
    """
    Parse command-line arguments for input and output video, VMAF model path, and CRF value.
    """
    parser = argparse.ArgumentParser(description="Process a video, extract metrics, and update CSV.")

    # Define command-line arguments
    parser.add_argument('input_video', type=str, help="Path to the input video file.")
    parser.add_argument('output_video', type=str, help="Path to the output video file.")
    parser.add_argument('--vmaf_model_path', type=str, default=None, help="Path to the VMAF model file. If not provided, FFmpeg's default model will be used.")
    parser.add_argument('--crf', type=int, default=23, help="CRF (Constant Rate Factor) value for FFmpeg encoding. Default is 23.")

    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse the command-line arguments
        args = parse_arguments()

        # Get video info to obtain width and height
        bitrate, resolution, frame_rate, width, height = get_video_info(args.input_video)

        # Calculate the aspect ratio
        aspect_ratio = calculate_aspect_ratio(width, height)
        logger.info(f"Video Aspect Ratio: {aspect_ratio[0]}:{aspect_ratio[1]}")

        # Choose base heights for testing
        base_heights = [90, 180, 360, 720]  # Adjust as needed

        # Generate resolutions based on aspect ratio
        resolutions_to_test = generate_resolutions(aspect_ratio, base_heights)
        logger.info(f"Resolutions to test: {resolutions_to_test}")

        # Select a resolution (you can adjust this as needed)
        selected_width, selected_height = resolutions_to_test[1]  # Adjust based on your observations
        logger.info(f"Selected resolution for processing: {selected_width}x{selected_height}")

        # Define frame interval
        frame_interval = 10  # Process every 10th frame

        # Proceed with the rest of the processing using the selected resolution
        process_video_and_extract_metrics(
            input_video=args.input_video,
            crf=args.crf,
            output_video=args.output_video,
            vmaf_model_path=args.vmaf_model_path,  # May be None
            resize_width=selected_width,
            resize_height=selected_height,
            frame_interval=frame_interval
        )

        logger.info(f"Processing completed for video: {args.input_video}")
    except Exception as e:
        logger.error("An error occurred: %s", e)
        sys.exit(1)
