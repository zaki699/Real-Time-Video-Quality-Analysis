import concurrent.futures
import argparse
import re
import subprocess
import cv2
import joblib
import numpy as np
import pandas as pd
import os
import sys
import logging
import tempfile
import shutil
import time
import math
from tqdm import tqdm  # Progress bar

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

# Process frame complexity by calculating motion complexity between two consecutive frames
def process_frame_complexity(frame, prev_frame):
    if frame is None or prev_frame is None:
        logger.warning("Skipping empty frame.")
        return 0.0  # Return a default value for empty frames
    
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow between consecutive frames
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of motion vectors (motion magnitude)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_motion = np.mean(mag)

    return avg_motion

# Advanced Motion Complexity calculation with parallel frame processing
def calculate_advanced_motion_complexity(video_path, frame_interval=10, min_frames_for_parallel=50):
    cap = cv2.VideoCapture(video_path)
    total_motion = []
    frames = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_count in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval != 0:
                continue
            frames.append((frame, prev_frame if frame_count > 0 else None))
            prev_frame = frame
    finally:
        cap.release()

    # Decide whether to parallelize or not based on the number of frames
    if len(frames) > min_frames_for_parallel:
        logger.info("Parallelizing motion complexity processing.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            complexities = list(executor.map(lambda f: process_frame_complexity(f[0], f[1]), frames))
    else:
        logger.info("Sequential motion complexity processing.")
        complexities = [process_frame_complexity(f[0], f[1]) for f in frames]

    smoothed_motion = smooth_data(complexities)

    if smoothed_motion.size > 0:  # Use size to check if the array is non-empty
        return np.mean(smoothed_motion)
    else:
        return 0.0

# SIFT and ORB are advanced feature detectors used to identify key points and descriptors in images.
#  These can be used to evaluate frame complexity by detecting and counting the number of keypoints in each frame. 
def calculate_orb_feature_complexity(video_path, frame_interval=10, resize_width=64, resize_height=64):
    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create()  # Create an ORB detector
    total_keypoints = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            # Resize frame for faster processing
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Detect ORB keypoints
            keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

            # Complexity can be measured by the number of keypoints detected
            total_keypoints.append(len(keypoints))

    finally:
        cap.release()

    return np.mean(total_keypoints) if total_keypoints else 0.0
    
# A color histogram represents the distribution of color intensities in an image.
#  You can analyze the distribution of colors across the frames to gauge the complexity
#  of scenes with varying color intensity and diversity.
def calculate_color_histogram_complexity(video_path, frame_interval=10, resize_width=64, resize_height=64):
    cap = cv2.VideoCapture(video_path)
    total_histogram_complexity = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            # Resize frame for faster processing
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            # Calculate histogram for each channel (B, G, R)
            hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])

            # Normalize histograms
            hist_b /= hist_b.sum()
            hist_g /= hist_g.sum()
            hist_r /= hist_r.sum()

            # Calculate entropy as complexity measure
            hist_entropy = - (np.sum(hist_b * np.log2(hist_b + 1e-8)) + 
                              np.sum(hist_g * np.log2(hist_g + 1e-8)) + 
                              np.sum(hist_r * np.log2(hist_r + 1e-8)))

            total_histogram_complexity.append(hist_entropy)

    finally:
        cap.release()

    return np.mean(total_histogram_complexity) if total_histogram_complexity else 0.0

# DCT complexity with parallel frame processing
def calculate_dct_scene_complexity(video_path, resize_width, resize_height, frame_interval=10, min_frames_for_parallel=50):
    cap = cv2.VideoCapture(video_path)
    total_dct_energy = []
    frames = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) > min_frames_for_parallel:
        logger.info("Parallelizing DCT complexity processing.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dct_energies = list(executor.map(lambda f: process_dct_frame(f, resize_width, resize_height), frames))
    else:
        logger.info("Sequential DCT complexity processing.")
        dct_energies = [process_dct_frame(f, resize_width, resize_height) for f in frames]

    smoothed_dct = smooth_data(dct_energies)

    if smoothed_dct.size > 0:
        return np.mean(smoothed_dct)
    else:
        return 0.0

def process_dct_frame(frame, resize_width, resize_height):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))  # Resize for DCT calculation
    dct_frame = cv2.dct(np.float32(gray_frame))
    return np.sum(dct_frame ** 2)

# Temporal DCT complexity
def calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    total_temporal_dct_energy = []
    prev_frame_dct = None
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0.0

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

    if smoothed_temporal_dct.size > 0:
        return np.mean(smoothed_temporal_dct)
    else:
        return 0.0

# Histogram complexity with parallel frame processing
def calculate_histogram_complexity(video_path, resize_width, resize_height, frame_interval=10, min_frames_for_parallel=50):
    cap = cv2.VideoCapture(video_path)
    total_entropy = []
    frames = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) > min_frames_for_parallel:
        logger.info("Parallelizing histogram complexity processing.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            entropies = list(executor.map(lambda f: process_histogram_frame(f, resize_width, resize_height), frames))
    else:
        logger.info("Sequential histogram complexity processing.")
        entropies = [process_histogram_frame(f, resize_width, resize_height) for f in frames]

    smoothed_entropy = smooth_data(entropies)

    if smoothed_entropy.size > 0:
        return np.mean(smoothed_entropy)
    else:
        return 0.0

def process_histogram_frame(frame, resize_width, resize_height):
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize the histogram
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return entropy

# Edge Detection complexity with parallel frame processing
def calculate_edge_detection_complexity(video_path, resize_width, resize_height, frame_interval=10, min_frames_for_parallel=50):
    cap = cv2.VideoCapture(video_path)
    total_edges = []
    frames = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) > min_frames_for_parallel:
        logger.info("Parallelizing edge detection complexity processing.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            edges = list(executor.map(lambda f: process_edge_frame(f, resize_width, resize_height), frames))
    else:
        logger.info("Sequential edge detection complexity processing.")
        edges = [process_edge_frame(f, resize_width, resize_height) for f in frames]

    smoothed_edges = smooth_data(edges)

    if smoothed_edges.size > 0:
        return np.mean(smoothed_edges)
    else:
        return 0.0

def process_edge_frame(frame, resize_width, resize_height):
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200)
    return np.sum(edges > 0)  # Count the number of edge pixels

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
    histogram_complexity = calculate_color_histogram_complexity(video_path, frame_interval, resize_width, resize_height)
    
    logger.info("Calculating edge detection complexity...")
    edge_detection_complexity = calculate_edge_detection_complexity(video_path, resize_width, resize_height, frame_interval)
    
    logger.info("Calculating ORB feature complexity...")
    orb_feature_complexity = calculate_orb_feature_complexity(video_path, frame_interval, resize_width, resize_height)

    logger.info(f"Advanced Motion Complexity: {advanced_motion_complexity:.2f}")
    logger.info(f"DCT Complexity: {dct_complexity:.2f}")
    logger.info(f"Temporal DCT Complexity: {temporal_dct_complexity:.2f}")
    logger.info(f"Histogram Complexity: {histogram_complexity:.2f}")
    logger.info(f"Edge Detection Complexity: {edge_detection_complexity:.2f}")
    logger.info(f"ORB Feature Complexity: {orb_feature_complexity:.2f}")

    # Define min and max values for normalization (these values should be based on your dataset)
    metric_min_values = {
        'Advanced Motion Complexity': 0.0,
        'DCT Complexity': 1e6,
        'Temporal DCT Complexity': 0.0,
        'Histogram Complexity': 0.0,
        'Edge Detection Complexity': 0.0,
        'ORB Feature Complexity': 0.0
    }

    metric_max_values = {
        'Advanced Motion Complexity': 10.0,
        'DCT Complexity': 5e7,
        'Temporal DCT Complexity': 1e7,
        'Histogram Complexity': 8.0,
        'Edge Detection Complexity': resize_width * resize_height,  # Maximum possible edges
        'ORB Feature Complexity': 5000  # Adjust this value based on expected keypoints
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
    normalized_metrics['ORB Feature Complexity'] = min_max_normalize(
        orb_feature_complexity, metric_min_values['ORB Feature Complexity'], metric_max_values['ORB Feature Complexity'])

    # Define weights for each metric (adjust these weights as needed)
    weights = {
        'Advanced Motion Complexity': 0.20,
        'DCT Complexity': 0.20,
        'Temporal DCT Complexity': 0.20,
        'Histogram Complexity': 0.15,
        'Edge Detection Complexity': 0.10,
        'ORB Feature Complexity': 0.15
    }

    # Calculate weighted average
    total_weight = sum(weights.values())
    weighted_sum = sum(normalized_metrics[metric] * weights[metric] for metric in normalized_metrics)
    overall_complexity = weighted_sum / total_weight

    logger.info(f"Normalized Metrics: {normalized_metrics}")
    logger.info(f"Overall Scene Complexity (Weighted Average): {overall_complexity:.4f}")

    return overall_complexity

def calculate_temporal_dct_frame(frame, resize_width, resize_height, prev_frame_dct=None):
    """
    Calculate temporal DCT complexity for a single frame by comparing it to the previous frame.
    
    Args:
        frame (np.array): The current video frame.
        resize_width (int): Target width for resizing the frame.
        resize_height (int): Target height for resizing the frame.
        prev_frame_dct (np.array): The DCT of the previous frame for comparison.
    
    Returns:
        float: Temporal DCT complexity (the difference between current and previous frame DCT).
    """
    # Resize and convert the frame to grayscale
    gray_frame = cv2.cvtColor(cv2.resize(frame, (resize_width, resize_height)), cv2.COLOR_BGR2GRAY)

    # Perform DCT on the current frame
    curr_frame_dct = cv2.dct(np.float32(gray_frame))

    # If there's no previous frame, return 0 as we can't calculate the difference
    if prev_frame_dct is None:
        return 0.0

    # Calculate the temporal DCT complexity (difference between current and previous DCT)
    temporal_dct_diff = np.sum((curr_frame_dct - prev_frame_dct) ** 2)

    return temporal_dct_diff

def calculate_histogram_complexity_frame(frame, resize_width, resize_height):
    """
    Calculate histogram complexity for a single frame.
    
    Args:
        frame (np.array): The current video frame.
        resize_width (int): Target width for resizing the frame.
        resize_height (int): Target height for resizing the frame.
    
    Returns:
        float: Entropy value for the frame's histogram.
    """
    # Resize frame
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize the histogram

    # Calculate the entropy (complexity) of the histogram
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    return entropy


# Function to run FFmpeg to compute PSNR, SSIM, and VMAF
def run_ffmpeg_metrics(reference_video, distorted_video, vmaf_model_path=None):
    # Get the total video duration (in seconds) using ffprobe
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        reference_video
    ]
    
    process = subprocess.Popen(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    total_duration = float(stdout.strip()) if stdout.strip() else None

    # Construct FFmpeg command
    ffmpeg_path = 'ffmpeg'
    log_dir = os.getcwd()
    psnr_log = os.path.join(log_dir, 'psnr.log')
    ssim_log = os.path.join(log_dir, 'ssim.log')
    vmaf_log = os.path.join(log_dir, 'vmaf.json')

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

    # Call FFmpeg with progress bar
    logger.info(f"Running FFmpeg to compute PSNR, SSIM, and VMAF metrics...")
    run_with_progress(cmd, total_duration, description="Calculating Metrics")

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

def run_with_progress(cmd, total_duration=None, description="Processing"):
    """
    Run a subprocess command with a progress bar.
    
    Parameters:
    - cmd: The subprocess command to run (list).
    - total_duration: Estimated total duration of the process in seconds (optional).
    - description: Description to display on the progress bar (optional).
    
    Returns:
    - stdout and stderr from the subprocess.
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    progress_bar = tqdm(total=total_duration, desc=description, unit="s", dynamic_ncols=True)

    start_time = time.time()
    
    while process.poll() is None:
        # FFmpeg outputs progress updates to stderr, so we check that
        output = process.stderr.readline()
        
        # We can parse FFmpeg progress info here (for example, extracting time duration)
        if "time=" in output:
            match = re.search(r"time=(\d+:\d+:\d+.\d+)", output)
            if match:
                current_time = match.group(1)
                h, m, s = map(float, current_time.split(':'))
                elapsed_time = h * 3600 + m * 60 + s
                
                # Update the progress bar with the elapsed time
                progress_bar.update(min(elapsed_time - progress_bar.n, total_duration - progress_bar.n))
        
        time.sleep(0.1)  # Sleep briefly to allow the process to run

    process.communicate()  # Wait for the process to complete
    progress_bar.close()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    return process.stdout, process.stderr



def process_video_and_extract_metrics(input_video, crf, output_video, vmaf_model_path, resize_width, resize_height, frame_interval=10):
    # Validate input video
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"The input video file {input_video} does not exist.")
    
    # Get the total video duration (in seconds) using ffprobe
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_video
    ]
    
    process = subprocess.Popen(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    total_duration = float(stdout.strip()) if stdout.strip() else None

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
        
        # Call FFmpeg with progress bar
        logger.info(f"Encoding video with CRF {crf}...")
        run_with_progress(encode_cmd, total_duration, description="Encoding Video")

        # Run FFmpeg to compute metrics between the original and encoded videos
        run_ffmpeg_metrics(input_video, encoded_video, vmaf_model_path)

        # Extract bitrate, resolution, and frame rate from the input video
        bitrate, resolution, frame_rate, _, _ = get_video_info(input_video)


        # Step 3: Extract actual metrics from logs generated by FFmpeg (PSNR, SSIM, VMAF)
        logger.info("Extracting actual metrics from FFmpeg logs...")
        actual_metrics = extract_metrics_from_logs(
            psnr_log='psnr.log',
            ssim_log='ssim.log',
            vmaf_log='vmaf.json',
            video_file=input_video,
            crf=crf,
            bitrate=bitrate,
            resolution=resolution,
            frame_rate=frame_rate,
            resize_width=resize_width,
            resize_height=resize_height,
            frame_interval=frame_interval
        )

        # Step 4: Log and save combined metrics
        logger.info(f"Combined metrics extracted: {actual_metrics}")
        update_csv(actual_metrics, csv_file='video_quality_data.csv')

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg process failed: {e}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
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
