import argparse
import re
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import pandas as pd
import os
import sys
import logging
import time
import math

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

# Scene cut detection function
def detect_scene_cuts(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    scene_cuts = []
    prev_hist = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate histogram for the current frame
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            # Compute the correlation between histograms
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            # If the correlation is below the threshold, consider it a scene cut
            if hist_diff < threshold:
                scene_cuts.append(frame_idx)
        else:
            # First frame is always a scene cut
            scene_cuts.append(frame_idx)

        prev_hist = hist
        frame_idx += 1

    cap.release()
    return scene_cuts

# Advanced Motion Complexity calculation with scene cuts
def calculate_advanced_motion_complexity(video_path):
    cap = cv2.VideoCapture(video_path)
    total_motion = []
    motion_changes = []

    # Detect scene cuts
    scene_cuts = detect_scene_cuts(video_path)

    try:
        prev_gray = None
        prev_motion = None

        for frame_idx in scene_cuts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Calculate Optical Flow between consecutive frames
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)

                # Calculate the magnitude of motion vectors (motion magnitude)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_motion = np.mean(mag)
                total_motion.append(avg_motion)

                if prev_motion is not None:
                    motion_change = abs(avg_motion - prev_motion)
                    motion_changes.append(motion_change)

                prev_motion = avg_motion
            else:
                # For the first frame
                total_motion.append(0)
                prev_motion = 0

            prev_gray = curr_gray

    finally:
        cap.release()

    # Smooth the motion complexity values
    smoothed_motion = smooth_data(total_motion)

    # Calculate temporal consistency by averaging changes in motion between frames
    avg_temporal_consistency = np.mean(motion_changes) if len(motion_changes) > 0 else 0.0

    # Return the average motion magnitude and temporal consistency as the final motion complexity score
    return np.mean(smoothed_motion) + avg_temporal_consistency

# DCT complexity with scene cuts
def calculate_dct_scene_complexity(video_path, resize_width, resize_height):
    cap = cv2.VideoCapture(video_path)
    total_dct_energy = []

    # Detect scene cuts
    scene_cuts = detect_scene_cuts(video_path)

    try:
        for frame_idx in scene_cuts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))

            dct_frame = cv2.dct(np.float32(gray_frame))
            energy = np.sum(dct_frame ** 2)
            total_dct_energy.append(energy)
    finally:
        cap.release()

    # Smooth the DCT complexity values
    smoothed_dct = smooth_data(total_dct_energy)
    return np.mean(smoothed_dct) if len(smoothed_dct) > 0 else 0.0

# Temporal DCT complexity with scene cuts
def calculate_temporal_dct(video_path, resize_width, resize_height):
    cap = cv2.VideoCapture(video_path)
    total_temporal_dct_energy = []
    prev_frame_dct = None

    # Detect scene cuts
    scene_cuts = detect_scene_cuts(video_path)

    try:
        for frame_idx in scene_cuts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
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

    # Smooth the Temporal DCT complexity values
    smoothed_temporal_dct = smooth_data(total_temporal_dct_energy)
    return np.mean(smoothed_temporal_dct) if len(smoothed_temporal_dct) > 0 else 0.0

# Histogram complexity with scene cuts
def calculate_histogram_complexity(video_path, resize_width, resize_height):
    cap = cv2.VideoCapture(video_path)
    total_entropy = []

    # Detect scene cuts
    scene_cuts = detect_scene_cuts(video_path)

    try:
        for frame_idx in scene_cuts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize frame
            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalize the histogram
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            total_entropy.append(entropy)
    finally:
        cap.release()

    # Smooth the histogram complexity values
    smoothed_entropy = smooth_data(total_entropy)
    return np.mean(smoothed_entropy) if len(smoothed_entropy) > 0 else 0.0

# Edge Detection complexity with scene cuts
def calculate_edge_detection_complexity(video_path, resize_width, resize_height):
    cap = cv2.VideoCapture(video_path)
    total_edges = []

    # Detect scene cuts
    scene_cuts = detect_scene_cuts(video_path)

    try:
        for frame_idx in scene_cuts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize frame
            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_frame, 100, 200)
            edge_count = np.sum(edges > 0)  # Count the number of edge pixels
            total_edges.append(edge_count)
    finally:
        cap.release()

    # Smooth the edge complexity values
    smoothed_edges = smooth_data(total_edges)
    return np.mean(smoothed_edges) if len(smoothed_edges) > 0 else 0.0

# Final average scene complexity with adjustable resizing resolution
def calculate_average_scene_complexity(video_path, resize_width, resize_height):
    logger.info("Calculating advanced motion complexity...")
    advanced_motion_complexity = calculate_advanced_motion_complexity(video_path)
    logger.info("Calculating DCT scene complexity...")
    dct_complexity = calculate_dct_scene_complexity(video_path, resize_width, resize_height)
    logger.info("Calculating temporal DCT complexity...")
    temporal_dct_complexity = calculate_temporal_dct(video_path, resize_width, resize_height)
    logger.info("Calculating histogram complexity...")
    histogram_complexity = calculate_histogram_complexity(video_path, resize_width, resize_height)
    logger.info("Calculating edge detection complexity...")
    edge_detection_complexity = calculate_edge_detection_complexity(video_path, resize_width, resize_height)

    logger.info(f"Advanced Motion Complexity: {advanced_motion_complexity:.2f}")
    logger.info(f"DCT Complexity: {dct_complexity:.2f}")
    logger.info(f"Temporal DCT Complexity: {temporal_dct_complexity:.2f}")
    logger.info(f"Histogram Complexity: {histogram_complexity:.2f}")
    logger.info(f"Edge Detection Complexity: {edge_detection_complexity:.2f}")

    # Calculate average complexity across all methods
    average_complexity = (advanced_motion_complexity + dct_complexity + temporal_dct_complexity +
                          histogram_complexity + edge_detection_complexity) / 5
    return average_complexity

def run_ffmpeg(input_video, crf, output_video, vmaf_model_path):
    # Validate VMAF model path
    if not os.path.isfile(vmaf_model_path):
        raise FileNotFoundError(f"VMAF model not found at {vmaf_model_path}")

    # Construct FFmpeg command
    ffmpeg_path = 'ffmpeg'
    # Ensure the logs are stored in a specific directory
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
    vmaf_model_path_escaped = quote_path(vmaf_model_path)

    cmd = [
        ffmpeg_path,
        '-i', input_video,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', 'medium',
        '-lavfi', (
            f"[0:v]split=3[psnr_in][ssim_in][vmaf_in];"
            f"[psnr_in]psnr=stats_file={psnr_log_escaped};"
            f"[ssim_in]ssim=stats_file={ssim_log_escaped};"
            f"[vmaf_in]libvmaf="
            f"log_path={vmaf_log_escaped}:log_fmt=json"
        ),
        '-f', 'null',
        '-'
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print FFmpeg output for debugging
    logger.debug("FFmpeg Output: %s", stdout.decode())
    logger.debug("FFmpeg Error Output: %s", stderr.decode())

    if process.returncode != 0:
        logger.error("FFmpeg process failed with return code %s", process.returncode)
        logger.error("FFmpeg stderr: %s", stderr.decode())
        raise RuntimeError(f"FFmpeg process failed with return code {process.returncode}")
    return stdout, stderr

def extract_metrics_from_logs(psnr_log, ssim_log, vmaf_log, video_file, crf, bitrate, resolution, frame_rate, resize_width, resize_height):
    # Initialize metrics
    psnr = None
    ssim = None
    vmaf = None

    # Extract PSNR value from the log file
    if os.path.isfile(psnr_log):
        with open(psnr_log) as f:
            content = f.read()
            match = re.search(r'average:(\s*\d+\.\d+)', content)
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
    scene_complexity = calculate_average_scene_complexity(video_file, resize_width, resize_height)

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

def run_ffmpeg_metrics(reference_video, distorted_video, vmaf_model_path):
    # Validate VMAF model path
    if not os.path.isfile(vmaf_model_path):
        raise FileNotFoundError(f"VMAF model not found at {vmaf_model_path}")

    # Construct FFmpeg command
    ffmpeg_path = 'ffmpeg'
    log_dir = os.getcwd()
    psnr_log = os.path.join(log_dir, 'psnr.log')
    ssim_log = os.path.join(log_dir, 'ssim.log')
    vmaf_log = os.path.join(log_dir, 'vmaf.json')

    cmd = [
        ffmpeg_path,
        '-i', distorted_video,
        '-i', reference_video,
        '-lavfi', (
            f"[0:v][1:v]psnr=stats_file={psnr_log};"
            f"[0:v][1:v]ssim=stats_file={ssim_log};"
            f"[0:v][1:v]libvmaf="
            f"log_path={vmaf_log}:log_fmt=json"
        ),
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

    return stdout, stderr

def process_video_and_extract_metrics(input_video, crf, output_video, vmaf_model_path, resize_width, resize_height):
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
        run_ffmpeg_metrics(input_video, encoded_video, vmaf_model_path)

        # Extract bitrate, resolution, and frame rate from the input video
        bitrate, resolution, frame_rate, _, _ = get_video_info(input_video)

        # Extract metrics from the generated logs (PSNR, SSIM, VMAF)
        metrics = extract_metrics_from_logs(
            psnr_log='psnr.log',
            ssim_log='ssim.log',
            vmaf_log='vmaf.json',
            video_file=input_video,
            crf=crf,
            bitrate=bitrate,
            resolution=resolution,
            frame_rate=frame_rate,
            resize_width=resize_width,
            resize_height=resize_height
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
    parser.add_argument('vmaf_model_path', type=str, help="Path to the VMAF model file.")
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

        # Select a resolution (you can still test resolutions if desired)
        selected_width, selected_height = resolutions_to_test[1]  # Adjust based on your observations
        logger.info(f"Selected resolution for processing: {selected_width}x{selected_height}")

        # Proceed with the rest of the processing using the selected resolution
        process_video_and_extract_metrics(
            input_video=args.input_video,
            crf=args.crf,
            output_video=args.output_video,
            vmaf_model_path=args.vmaf_model_path,
            resize_width=selected_width,
            resize_height=selected_height
        )

        logger.info(f"Processing completed for video: {args.input_video}")
    except Exception as e:
        logger.error("An error occurred: %s", e)
        sys.exit(1)
