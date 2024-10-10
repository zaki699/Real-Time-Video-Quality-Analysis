import functools
import cv2
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing
from itertools import tee


logger = logging.getLogger(__name__)

# Extract frame timestamps from the video
def extract_frame_timestamps(video_path, frame_interval=10):
    """
    Extracts frame timestamps from the video file.

    Parameters:
        video_path (str): The path to the video file.
        frame_interval (int): The interval at which frames should be processed.

    Returns:
        list: A list of timestamps for the frames in milliseconds.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_timestamps = []

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []

        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Capture the timestamp for the current frame
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamps.append(timestamp)

            frame_count += 1

    finally:
        cap.release()

    return frame_timestamps


# Read frames from video with optional frame interval and return pairs of consecutive frames
def read_frame_pairs(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                if prev_frame is not None:
                    frames.append((frame, prev_frame))  # Append the current frame and the previous frame as a pair
                prev_frame = frame

    finally:
        cap.release()

    return frames


# Exponential smoothing using pandas
def smooth_data(data, alpha=0.8):
    return pd.Series(data).ewm(alpha=alpha).mean().to_numpy()

# Batch processing for parallel tasks
def process_in_batches(frames, process_func, num_workers, batch_size=100, **kwargs):
    """
    Process frames in batches using ProcessPoolExecutor.
    Args:
    - frames: list of frames to process.
    - process_func: the function to apply to each batch.
    - num_workers: number of worker processes to use.
    - batch_size: number of frames per batch.
    - kwargs: additional arguments to pass to the process function.
    Returns:
    - List of processed results.
    """
    results = []
    with Executor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(frames), batch_size), desc="Batch Processing"):
            batch = frames[i:i+batch_size]
            process_with_args = functools.partial(process_func, **kwargs)  # Bind additional arguments
            results.extend(executor.map(process_with_args, batch))
    return results

# Calculate average scene complexity based on multiple metrics
def calculate_average_scene_complexity(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8, num_workers=None, batch_size=100):
    frame_pairs = read_frame_pairs(video_path, frame_interval)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    logger.info("Calculating advanced motion complexity...")
    motion_complexity = process_in_batches(frame_pairs, process_frame_complexity, num_workers, batch_size)
    smoothed_motion_complexity = smooth_data(motion_complexity, smoothing_factor)

    # Rest of the processing remains unchanged because they work on single frames
    frames = [pair[0] for pair in frame_pairs]  # Only take the first frame from each pair for other metrics

    logger.info("Calculating DCT scene complexity...")
    dct_complexity = process_in_batches(frames, functools.partial(process_dct_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_dct_complexity = smooth_data(dct_complexity, smoothing_factor)

    logger.info("Calculating temporal DCT complexity...")
    temporal_dct_complexity = calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval, smoothing_factor)

    logger.info("Calculating histogram complexity...")
    histogram_complexity = process_in_batches(frames, functools.partial(process_histogram_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_histogram_complexity = smooth_data(histogram_complexity, smoothing_factor)

    logger.info("Calculating edge detection complexity...")
    edge_detection_complexity = process_in_batches(frames, functools.partial(process_edge_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_edge_detection_complexity = smooth_data(edge_detection_complexity, smoothing_factor)

    logger.info("Calculating ORB feature complexity...")
    orb_feature_complexity = process_in_batches(frames, process_orb_frame_for_parallel, num_workers, batch_size)
    smoothed_orb_feature_complexity = smooth_data(orb_feature_complexity, smoothing_factor)

    logger.info("Calculating color histogram complexity...")
    color_histogram_complexity = process_in_batches(frames, functools.partial(process_color_histogram_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_color_histogram_complexity = smooth_data(color_histogram_complexity, smoothing_factor)

      # Calculate the frame rate variation using the timestamps from the frames
    frame_timestamps = extract_frame_timestamps(video_path, frame_interval)
    timestamp_pairs = list(zip(frame_timestamps[:-1], frame_timestamps[1:]))  # Create pairs of consecutive timestamps
    framerate_variation = process_in_batches(timestamp_pairs, process_frame_interval_for_parallel, num_workers, batch_size)
    smoothed_framerate_variation = smooth_data(framerate_variation, smoothing_factor)


    # Logging smoothed values
    logger.info(f"Smoothed Advanced Motion Complexity: {np.mean(smoothed_motion_complexity):.2f}")
    logger.info(f"Smoothed DCT Complexity: {np.mean(smoothed_dct_complexity):.2f}")
    logger.info(f"Smoothed Temporal DCT Complexity: {temporal_dct_complexity:.2f}")
    logger.info(f"Smoothed Histogram Complexity: {np.mean(smoothed_histogram_complexity):.2f}")
    logger.info(f"Smoothed Edge Detection Complexity: {np.mean(smoothed_edge_detection_complexity):.2f}")
    logger.info(f"Smoothed ORB Feature Complexity: {np.mean(smoothed_orb_feature_complexity):.2f}")
    logger.info(f"Smoothed Color Histogram Complexity: {np.mean(smoothed_color_histogram_complexity):.2f}")
    logger.info(f"Smoothed Frame Rate Variation: {np.mean(smoothed_framerate_variation):.2f}")

    return (
        np.mean(smoothed_motion_complexity),
        np.mean(smoothed_dct_complexity),
        temporal_dct_complexity,  # Already smoothed inside
        np.mean(smoothed_histogram_complexity),
        np.mean(smoothed_edge_detection_complexity),
        np.mean(smoothed_orb_feature_complexity),
        np.mean(smoothed_color_histogram_complexity),
        np.mean(smoothed_framerate_variation)
    )

# Helper function to calculate frame rate between two timestamps
def process_frame_interval_for_parallel(timestamps):
    """
    A helper function to calculate the frame rate from two timestamps.

    Parameters:
        timestamps (tuple): A tuple containing (prev_timestamp, curr_timestamp).

    Returns:
        float: The frame rate between two frames.
    """
    prev_timestamp, curr_timestamp = timestamps
    frame_interval = (curr_timestamp - prev_timestamp) / 1000.0  # Convert ms to s
    if frame_interval > 0:
        return 1.0 / frame_interval  # Frame rate
    return 0.0  # Return 0 if the frame interval is invalid

# Main function to calculate frame rate variation
def calculate_framerate_variation(frame_timestamps, smoothing_factor=0.8):
    """
    Calculate frame rate variation using timestamps from frames.

    Parameters:
        frame_timestamps (list): List of timestamps in milliseconds for each frame.
        smoothing_factor (float): Smoothing factor for exponential smoothing.

    Returns:
        dict: Dictionary containing smoothed frame rate statistics.
    """
    frame_rates = []
    
    # Calculate frame rates using consecutive frame timestamps
    for i in range(1, len(frame_timestamps)):
        frame_rate = process_frame_interval_for_parallel(frame_timestamps[i - 1], frame_timestamps[i])
        if frame_rate > 0:
            frame_rates.append(frame_rate)
    
    if len(frame_rates) > 0:
        # Apply exponential smoothing to the frame rates
        frame_rate_series = pd.Series(frame_rates)
        smoothed_frame_rates = frame_rate_series.ewm(alpha=smoothing_factor).mean().to_list()

        # Compute statistics for the smoothed frame rates
        average_framerate = np.mean(smoothed_frame_rates)
        min_framerate = np.min(smoothed_frame_rates)
        max_framerate = np.max(smoothed_frame_rates)

        framerate_variation = {
            'average_framerate': average_framerate,
            'min_framerate': min_framerate,
            'max_framerate': max_framerate,
            'smoothed_frame_rate_variation': smoothed_frame_rates
        }
        return framerate_variation
    else:
        return None
    
# Motion complexity using optical flow (updated for frame pairs)
def process_frame_complexity(frame_pair):
    frame, prev_frame = frame_pair
    if frame is None or prev_frame is None:
        logger.warning("Skipping frame due to missing previous frame.")
        return 0.0
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

# DCT Scene Complexity Calculation
def process_dct_frame(frame, resize_width, resize_height):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))
    dct_frame = cv2.dct(np.float32(gray_frame))
    return np.sum(dct_frame ** 2)

# ORB feature complexity calculation
def process_orb_frame_for_parallel(frame):
    orb = cv2.ORB_create()
    resized_frame = cv2.resize(frame, (64, 64))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    keypoints, _ = orb.detectAndCompute(gray_frame, None)
    return len(keypoints)

# Histogram complexity calculation
def process_histogram_frame(frame, resize_width, resize_height):
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return entropy

# Color Histogram Complexity Calculation
def process_color_histogram_frame(frame, resize_width, resize_height):
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])
    hist_b /= hist_b.sum()
    hist_g /= hist_g.sum()
    hist_r /= hist_r.sum()
    return - (np.sum(hist_b * np.log2(hist_b + 1e-8)) +
              np.sum(hist_g * np.log2(hist_g + 1e-8)) +
              np.sum(hist_r * np.log2(hist_r + 1e-8)))

# Edge Detection complexity calculation
def process_edge_frame(frame, resize_width, resize_height):
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    edge_pixels = cv2.Canny(gray_frame, 100, 200)
    return np.sum(edge_pixels > 0)

# Temporal DCT Complexity Calculation
def calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8):
    frames = read_frame_pairs(video_path, frame_interval)
    prev_gray_frame = None
    temporal_dct_energies = []

    for pair in frames:
        # Ensure the frame pair is valid
        frame, previous_frame = pair
        if frame is None or previous_frame is None or not isinstance(frame, (np.ndarray, cv2.UMat)):
            logger.error(f"Invalid frame detected: type={type(frame)}, skipping...")
            continue


        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Resize the frame to the specified dimensions
        gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))

        if prev_gray_frame is not None:
            # Process temporal DCT between consecutive frames
            temporal_energy = process_temporal_dct_frame(prev_gray_frame, gray_frame, resize_width, resize_height)
            temporal_dct_energies.append(temporal_energy)

        prev_gray_frame = gray_frame

    # Smooth and return the temporal DCT complexity
    smoothed_temporal_dct = smooth_data(temporal_dct_energies, smoothing_factor)
    return np.mean(smoothed_temporal_dct) if len(smoothed_temporal_dct) > 0 else 0.0

def process_temporal_dct_frame(prev_gray_frame, curr_gray_frame, resize_width, resize_height):
    prev_gray_frame = cv2.resize(prev_gray_frame, (resize_width, resize_height))
    curr_gray_frame = cv2.resize(curr_gray_frame, (resize_width, resize_height))
    prev_frame_dct = cv2.dct(np.float32(prev_gray_frame))
    curr_frame_dct = cv2.dct(np.float32(curr_gray_frame))
    return np.sum((curr_frame_dct - prev_frame_dct) ** 2)



