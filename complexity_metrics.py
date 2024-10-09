import functools
import cv2
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing


logger = logging.getLogger(__name__)

# Calculate average scene complexity based on multiple metrics
def calculate_average_scene_complexity(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8, min_frames_for_parallel=50):
    logger.info("Calculating advanced motion complexity...")
    advanced_motion_complexity = calculate_advanced_motion_complexity(video_path, frame_interval, smoothing_factor=smoothing_factor, min_frames_for_parallel=min_frames_for_parallel)
    logger.info("Calculating DCT scene complexity...")
    dct_complexity = calculate_dct_scene_complexity(video_path, resize_width, resize_height, frame_interval, smoothing_factor=smoothing_factor, min_frames_for_parallel=min_frames_for_parallel)
    logger.info("Calculating temporal DCT complexity...")
    temporal_dct_complexity = calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval, smoothing_factor=smoothing_factor)
    logger.info("Calculating histogram complexity...")
    histogram_complexity = calculate_histogram_complexity(video_path, resize_width, resize_height, frame_interval, smoothing_factor=smoothing_factor)
    logger.info("Calculating edge detection complexity...")
    edge_detection_complexity = calculate_edge_detection_complexity(video_path, resize_width, resize_height, frame_interval, min_frames_for_parallel=min_frames_for_parallel, smoothing_factor=smoothing_factor)
    logger.info("Calculating ORB feature complexity...")
    orb_feature_complexity = calculate_orb_feature_complexity(video_path, frame_interval, resize_width, resize_height, smoothing_factor=smoothing_factor)
    logger.info("Calculating color histogram complexity...")
    color_histogram_complexity = calculate_color_histogram_complexity(video_path, frame_interval, resize_width, resize_height, smoothing_factor=smoothing_factor)

    logger.info(f"Advanced Motion Complexity: {advanced_motion_complexity:.2f}")
    logger.info(f"DCT Complexity: {dct_complexity:.2f}")
    logger.info(f"Temporal DCT Complexity: {temporal_dct_complexity:.2f}")
    logger.info(f"Histogram Complexity: {histogram_complexity:.2f}")
    logger.info(f"Edge Detection Complexity: {edge_detection_complexity:.2f}")
    logger.info(f"ORB Feature Complexity: {orb_feature_complexity:.2f}")
    logger.info(f"Color Histogram Complexity: {color_histogram_complexity:.2f}")

    return (
        advanced_motion_complexity,
        dct_complexity,
        temporal_dct_complexity,
        histogram_complexity,
        edge_detection_complexity,
        orb_feature_complexity,
        color_histogram_complexity
    )


# Exponential smoothing using pandas
def smooth_data(data, alpha=0.8):
    return pd.Series(data).ewm(alpha=alpha).mean().to_numpy()

# Frame complexity calculation using optical flow
def process_frame_complexity(frame_pair):
    frame, prev_frame = frame_pair
    if frame is None or prev_frame is None:
        logging.warning("Skipping frame due to missing previous frame.")
        return 0.0
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

# Advanced motion complexity calculation
def calculate_advanced_motion_complexity(video_path, frame_interval=10, min_frames_for_parallel=50, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0
        for frame_count in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if frame is None:
                logging.warning(f"Skipping frame {frame_count}.")
                continue
            if frame_count % frame_interval != 0:
                continue
            if prev_frame is not None:
                frames.append((frame, prev_frame))
            prev_frame = frame
    finally:
        cap.release()

    num_workers = multiprocessing.cpu_count() // 2
    if len(frames) > min_frames_for_parallel:
        logging.info("Parallelizing motion complexity processing.")
        with Executor(max_workers=num_workers) as executor:
            complexities = list(executor.map(process_frame_complexity, frames))
    else:
        logging.info("Sequential motion complexity processing.")
        complexities = [process_frame_complexity(frame_pair) for frame_pair in frames]

    smoothed_motion = smooth_data(complexities, smoothing_factor)
    return np.mean(smoothed_motion) if smoothed_motion.size > 0 else 0.0

# ORB feature complexity calculation
def calculate_orb_feature_complexity(video_path, frame_interval=10, resize_width=64, resize_height=64, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create()
    total_keypoints = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            keypoints, _ = orb.detectAndCompute(gray_frame, None)
            total_keypoints.append(len(keypoints))

    finally:
        cap.release()

    smoothed_keypoints = smooth_data(total_keypoints, smoothing_factor)
    return np.mean(smoothed_keypoints) if smoothed_keypoints.size > 0 else 0.0

# Histogram complexity calculation
def calculate_histogram_complexity(video_path, frame_interval=10, resize_width=64, resize_height=64, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    total_entropy = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
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

    smoothed_entropy = smooth_data(total_entropy, smoothing_factor)
    return np.mean(smoothed_entropy) if smoothed_entropy.size > 0 else 0.0

# Color histogram complexity calculation
def calculate_color_histogram_complexity(video_path, frame_interval=10, resize_width=64, resize_height=64, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    total_histogram_complexity = []
    frame_count = 0

    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])

            hist_b /= hist_b.sum()
            hist_g /= hist_g.sum()
            hist_r /= hist_r.sum()

            hist_entropy = - (np.sum(hist_b * np.log2(hist_b + 1e-8)) +
                              np.sum(hist_g * np.log2(hist_g + 1e-8)) +
                              np.sum(hist_r * np.log2(hist_r + 1e-8)))

            total_histogram_complexity.append(hist_entropy)

    finally:
        cap.release()

    smoothed_histogram = smooth_data(total_histogram_complexity, smoothing_factor)
    return np.mean(smoothed_histogram) if smoothed_histogram.size > 0 else 0.0

# Edge detection frame processing
def process_edge_frame(frame, resize_width, resize_height):
    """
    Processes a frame to compute the number of edge pixels using the Canny edge detector.

    Parameters:
        frame (numpy.ndarray): The frame to process.
        resize_width (int): Width to resize the frame.
        resize_height (int): Height to resize the frame.

    Returns:
        int: Number of edge pixels in the frame.
    """
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200)
    return np.sum(edges > 0)  # Count the number of edge pixels

# Edge detection complexity calculation
def calculate_edge_detection_complexity(video_path, resize_width, resize_height, frame_interval=10, min_frames_for_parallel=50, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            frames.append(frame)
    finally:
        cap.release()

    num_workers = multiprocessing.cpu_count() // 2
    if len(frames) > min_frames_for_parallel:
        logging.info("Parallelizing edge detection complexity processing.")
        with Executor(max_workers=num_workers) as executor:
            # Use functools.partial to pass extra arguments to process_edge_frame
            process_func = functools.partial(process_edge_frame, resize_width=resize_width, resize_height=resize_height)
            edges = list(executor.map(process_func, frames))
    else:
        logging.info("Sequential edge detection complexity processing.")
        edges = [process_edge_frame(frame, resize_width, resize_height) for frame in frames]

    smoothed_edges = smooth_data(edges, smoothing_factor)
    return np.mean(smoothed_edges) if smoothed_edges.size > 0 else 0.0

# DCT (Discrete Cosine Transform) scene complexity calculation
def calculate_dct_scene_complexity(video_path, resize_width, resize_height, frame_interval=10, min_frames_for_parallel=50, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            frames.append(frame)
    finally:
        cap.release()

    # Determine the number of workers for parallel processing
    num_workers = multiprocessing.cpu_count() // 2
    if len(frames) > min_frames_for_parallel:
        logging.info("Parallelizing DCT complexity processing.")
        with Executor(max_workers=num_workers) as executor:
            # Use functools.partial to pass extra arguments to process_dct_frame
            process_func = functools.partial(process_dct_frame, resize_width=resize_width, resize_height=resize_height)
            dct_energies = list(executor.map(process_func, frames))
    else:
        logging.info("Sequential DCT complexity processing.")
        dct_energies = [process_dct_frame(frame, resize_width, resize_height) for frame in frames]

    # Smooth the results
    smoothed_dct = smooth_data(dct_energies, smoothing_factor)
    return np.mean(smoothed_dct) if smoothed_dct.size > 0 else 0.0

# DCT (Discrete Cosine Transform) scene complexity calculation
def process_dct_frame(frame, resize_width, resize_height):
    """
    Processes a frame to compute its DCT energy.

    Parameters:
        frame (numpy.ndarray): The frame to process.
        resize_width (int): Width to resize the frame.
        resize_height (int): Height to resize the frame.

    Returns:
        float: The DCT energy of the frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))  # Resize for DCT calculation
    dct_frame = cv2.dct(np.float32(gray_frame))
    return np.sum(dct_frame ** 2)



# Temporal DCT complexity calculation
def calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8):
    cap = cv2.VideoCapture(video_path)
    total_temporal_dct_energy = []
    prev_frame_dct = None
    frame_count = 0

    try:
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
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

    smoothed_temporal_dct = smooth_data(total_temporal_dct_energy, smoothing_factor)
    return np.mean(smoothed_temporal_dct) if smoothed_temporal_dct.size > 0 else 0.0
