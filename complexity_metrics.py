import numpy as np  # Always import NumPy for CPU operations
import cv2
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Executor
import functools
import multiprocessing

logger = logging.getLogger(__name__)

# Attempt to import CuPy for GPU processing (fallback to NumPy if unavailable)
try:
    import cupy as cp
    use_gpu = True
    logger.info("CuPy is available. Using GPU acceleration.")
except ImportError:
    cp = np  # Fallback to NumPy
    use_gpu = False
    logger.info("CuPy is not available. Using CPU processing.")


logger = logging.getLogger(__name__)

# Attempt to import CuPy for GPU processing (fallback to NumPy if unavailable)
try:
    import cupy as cp
    use_gpu = True
    logger.info("CuPy is available. Using GPU acceleration.")
except ImportError:
    cp = np  # Fallback to NumPy
    use_gpu = False
    logger.info("CuPy is not available. Using CPU processing.")

# Extract frame timestamps from the video
def extract_frame_timestamps(video_path, frame_interval=10):
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
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamps.append(timestamp)

            frame_count += 1
    finally:
        cap.release()

    return frame_timestamps


# Read frame pairs with a specified interval
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
                    frames.append((frame, prev_frame))  # Store current and previous frame as a pair
                prev_frame = frame
    finally:
        cap.release()

    return frames


# Exponential smoothing using GPU or CPU
def smooth_data(data, alpha=0.8):
    return pd.Series(data).ewm(alpha=alpha).mean().to_cupy() if use_gpu else pd.Series(data).ewm(alpha=alpha).mean().to_numpy()


# Process frames in batches with CuPy (GPU) or NumPy (CPU)
def process_in_batches(frames, process_func, num_workers, batch_size=100, **kwargs):
    results = []
    with Executor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(frames), batch_size), desc="Batch Processing"):
            batch = frames[i:i+batch_size]
            process_with_args = functools.partial(process_func, **kwargs)
            results.extend(executor.map(process_with_args, batch))
    return results


# Calculate scene complexity using various metrics (DCT, ORB, etc.)
def calculate_average_scene_complexity(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8, num_workers=None, batch_size=100):
    frame_pairs = read_frame_pairs(video_path, frame_interval)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    logger.info("Calculating advanced motion complexity...")
    motion_complexity = process_in_batches(frame_pairs, process_frame_complexity, num_workers, batch_size)
    smoothed_motion_complexity = smooth_data(motion_complexity, smoothing_factor)

    frames = [pair[0] for pair in frame_pairs]  # Use only the first frame for other metrics

    logger.info("Calculating DCT scene complexity...")
    dct_complexity = process_in_batches(frames, functools.partial(process_dct_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_dct_complexity = smooth_data(dct_complexity, smoothing_factor)

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

    logger.info("Calculating temporal DCT complexity...")
    temporal_dct_complexity = calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval, smoothing_factor)

    frame_timestamps = extract_frame_timestamps(video_path, frame_interval)
    timestamp_pairs = list(zip(frame_timestamps[:-1], frame_timestamps[1:]))  # Consecutive timestamps
    framerate_variation = process_in_batches(timestamp_pairs, process_frame_interval_for_parallel, num_workers, batch_size)
    smoothed_framerate_variation = smooth_data(framerate_variation, smoothing_factor)

    return (
        np.mean(smoothed_motion_complexity),
        np.mean(smoothed_dct_complexity),
        np.mean(smoothed_histogram_complexity),
        np.mean(smoothed_edge_detection_complexity),
        np.mean(smoothed_orb_feature_complexity),
        np.mean(smoothed_color_histogram_complexity),
        temporal_dct_complexity,  # Smoothed inside temporal DCT calculation
        np.mean(smoothed_framerate_variation)
    )


# Optical Flow using GPU or CPU
def process_frame_complexity(frame_pair):
    frame, prev_frame = frame_pair
    if frame is None or prev_frame is None:
        return 0.0

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if use_gpu:
        # Use CUDA-accelerated Optical Flow on GPU
        gpu_curr_gray = cv2.cuda_GpuMat()
        gpu_prev_gray = cv2.cuda_GpuMat()
        gpu_curr_gray.upload(curr_gray)
        gpu_prev_gray.upload(prev_gray)
        optical_flow = cv2.cuda_FarnebackOpticalFlow_create(5, 0.5, False)
        flow = optical_flow.calc(gpu_prev_gray, gpu_curr_gray, None)
    else:
        # Fallback to CPU-based optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)


# DCT Scene Complexity Calculation using GPU or CPU
def process_dct_frame(frame, resize_width, resize_height):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))
    if use_gpu:
        dct_frame = cv2.dct(cp.float32(gray_frame))  # Use CuPy for GPU
    else:
        dct_frame = cv2.dct(np.float32(gray_frame))  # Use NumPy for CPU
    return np.sum(dct_frame ** 2)


# ORB Feature Complexity using GPU or CPU
def process_orb_frame_for_parallel(frame):
    if use_gpu:
        orb = cv2.cuda_ORB_create()
        resized_frame = cv2.resize(frame, (64, 64))
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(resized_frame)
        gray_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        keypoints, _ = orb.detectAndCompute(gray_frame, None)
    else:
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        keypoints, _ = orb.detectAndCompute(gray_frame, None)
    
    return len(keypoints)


# Histogram complexity calculation using GPU or CPU
def process_histogram_frame(frame, resize_width, resize_height):
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    if use_gpu:
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cp.array(hist) / cp.sum(hist)
        entropy = -cp.sum(hist[hist > 0] * cp.log2(hist[hist > 0]))
    else:
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    return entropy


# Color Histogram Complexity using GPU or CPU
def process_color_histogram_frame(frame, resize_width, resize_height):
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    
    if use_gpu:
        # GPU processing with CuPy
        hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])
        hist_b = cp.array(hist_b) / cp.sum(hist_b)
        hist_g = cp.array(hist_g) / cp.sum(hist_g)
        hist_r = cp.array(hist_r) / cp.sum(hist_r)
        entropy = - (cp.sum(hist_b * cp.log2(hist_b + 1e-8)) +
                     cp.sum(hist_g * cp.log2(hist_g + 1e-8)) +
                     cp.sum(hist_r * cp.log2(hist_r + 1e-8)))
    else:
        # CPU processing with NumPy
        hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])
        hist_b = hist_b / hist_b.sum()
        hist_g = hist_g / hist_g.sum()
        hist_r = hist_r / hist_r.sum()
        entropy = - (np.sum(hist_b * np.log2(hist_b + 1e-8)) +
                     np.sum(hist_g * np.log2(hist_g + 1e-8)) +
                     np.sum(hist_r * np.log2(hist_r + 1e-8)))

    return entropy

def process_edge_frame(frame, resize_width, resize_height):
    """
    Calculate edge detection complexity using Canny edge detection algorithm.

    Parameters:
        frame: The input frame to process.
        resize_width: The target width to resize the frame.
        resize_height: The target height to resize the frame.

    Returns:
        float: The number of edge pixels in the frame.
    """
    # Resize the frame
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection (using GPU or CPU)
    if use_gpu:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(gray_frame)
        edge_pixels = cv2.cuda_CannyEdgeDetector().detect(gpu_frame)
        return cp.sum(edge_pixels.download() > 0)
    else:
        edge_pixels = cv2.Canny(gray_frame, 100, 200)
        return np.sum(edge_pixels > 0)
    
def calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8):
    frames = read_frame_pairs(video_path, frame_interval)
    prev_gray_frame = None
    temporal_dct_energies = []

    for pair in frames:
        frame, previous_frame = pair
        if frame is None or previous_frame is None:
            continue

        # Convert to grayscale and resize
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))

        if prev_gray_frame is not None:
            temporal_energy = process_temporal_dct_frame(prev_gray_frame, gray_frame, resize_width, resize_height)
            temporal_dct_energies.append(temporal_energy)

        prev_gray_frame = gray_frame

    # Smooth the data and return average temporal DCT complexity
    smoothed_temporal_dct = smooth_data(temporal_dct_energies, smoothing_factor)
    return np.mean(smoothed_temporal_dct) if len(smoothed_temporal_dct) > 0 else 0.0


def process_temporal_dct_frame(prev_gray_frame, curr_gray_frame, resize_width, resize_height):
    """
    Calculate the DCT difference between consecutive frames.
    """
    prev_gray_frame = cv2.resize(prev_gray_frame, (resize_width, resize_height))
    curr_gray_frame = cv2.resize(curr_gray_frame, (resize_width, resize_height))

    if use_gpu:
        prev_frame_dct = cv2.dct(cp.float32(prev_gray_frame))
        curr_frame_dct = cv2.dct(cp.float32(curr_gray_frame))
    else:
        prev_frame_dct = cv2.dct(np.float32(prev_gray_frame))
        curr_frame_dct = cv2.dct(np.float32(curr_gray_frame))

    return np.sum((curr_frame_dct - prev_frame_dct) ** 2)

def process_frame_interval_for_parallel(timestamps):
    """
    Calculate the frame rate interval between two consecutive timestamps.

    Parameters:
        timestamps (tuple): A tuple containing (prev_timestamp, curr_timestamp).

    Returns:
        float: The frame rate between two timestamps.
    """
    prev_timestamp, curr_timestamp = timestamps
    frame_interval = (curr_timestamp - prev_timestamp) / 1000.0  # Convert milliseconds to seconds

    if frame_interval > 0:
        return 1.0 / frame_interval  # Frame rate (in fps)
    return 0.0  # Return 0 if frame interval is invalid
