import argparse
import re
import subprocess
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
from tensorflow import keras



# Smoothing function for motion complexity and other metrics
def smooth_data(data, smoothing_factor=0.8):
    smoothed_data = np.zeros(len(data))
    smoothed_data[0] = data[0]  # Initialize with the first value

    for i in range(1, len(data)):
        smoothed_data[i] = (smoothing_factor * data[i] + 
                            (1 - smoothing_factor) * smoothed_data[i - 1])
    return smoothed_data

# Advanced Motion Complexity calculation with smoothing
def calculate_advanced_motion_complexity(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print("Error: Unable to read video.")
        return 0.0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_motion = []
    frame_count = 0
    motion_changes = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Optical Flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the magnitude of motion vectors (motion magnitude)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(mag)
        total_motion.append(avg_motion)
        
        # Calculate temporal consistency (change in motion magnitude between frames)
        if len(total_motion) > 1:
            motion_change = abs(total_motion[-1] - total_motion[-2])
            motion_changes.append(motion_change)
        
        prev_gray = curr_gray
        frame_count += 1
    
    cap.release()
    
    # Smooth the motion complexity values
    smoothed_motion = smooth_data(total_motion)
    
    # Calculate temporal consistency by averaging changes in motion between frames
    avg_temporal_consistency = np.mean(motion_changes) if len(motion_changes) > 0 else 0.0
    
    # Return the average motion magnitude and temporal consistency as the final motion complexity score
    return np.mean(smoothed_motion) + avg_temporal_consistency

# DCT complexity with smoothing
def calculate_dct_scene_complexity(video_path):
    cap = cv2.VideoCapture(video_path)
    total_dct_energy = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (64, 64))  # Resize for DCT calculation

        dct_frame = cv2.dct(np.float32(gray_frame))
        energy = np.sum(dct_frame ** 2)
        total_dct_energy.append(energy)
        frame_count += 1

    cap.release()
    
    # Smooth the DCT complexity values
    smoothed_dct = smooth_data(total_dct_energy)
    return np.mean(smoothed_dct) if len(smoothed_dct) > 0 else 0.0

# Temporal DCT complexity with smoothing
def calculate_temporal_dct(video_path):
    cap = cv2.VideoCapture(video_path)
    total_temporal_dct_energy = []
    frame_count = 0
    prev_frame_dct = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (64, 64))  # Resize for DCT calculation
        curr_frame_dct = cv2.dct(np.float32(gray_frame))

        if prev_frame_dct is not None:
            temporal_energy = np.sum((curr_frame_dct - prev_frame_dct) ** 2)
            total_temporal_dct_energy.append(temporal_energy)

        prev_frame_dct = curr_frame_dct
        frame_count += 1

    cap.release()
    
    # Smooth the Temporal DCT complexity values
    smoothed_temporal_dct = smooth_data(total_temporal_dct_energy)
    return np.mean(smoothed_temporal_dct) if len(smoothed_temporal_dct) > 0 else 0.0

# Histogram complexity with smoothing
def calculate_histogram_complexity(video_path):
    cap = cv2.VideoCapture(video_path)
    total_entropy = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize the histogram
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        total_entropy.append(entropy)
        frame_count += 1

    cap.release()
    
    # Smooth the histogram complexity values
    smoothed_entropy = smooth_data(total_entropy)
    return np.mean(smoothed_entropy) if len(smoothed_entropy) > 0 else 0.0

# Edge Detection complexity with smoothing
def calculate_edge_detection_complexity(video_path):
    cap = cv2.VideoCapture(video_path)
    total_edges = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200)
        edge_count = np.sum(edges > 0)  # Count the number of edge pixels
        total_edges.append(edge_count)

        frame_count += 1

    cap.release()
    
    # Smooth the edge complexity values
    smoothed_edges = smooth_data(total_edges)
    return np.mean(smoothed_edges) if len(smoothed_edges) > 0 else 0.0

# Final average scene complexity with smoothing for all components
def calculate_average_scene_complexity(video_path):
    advanced_motion_complexity = calculate_advanced_motion_complexity(video_path)
    dct_complexity = calculate_dct_scene_complexity(video_path)
    temporal_dct_complexity = calculate_temporal_dct(video_path)
    histogram_complexity = calculate_histogram_complexity(video_path)
    edge_detection_complexity = calculate_edge_detection_complexity(video_path)

    print(f"Advanced Motion Complexity: {advanced_motion_complexity:.2f}")
    print(f"DCT Complexity: {dct_complexity:.2f}")
    print(f"Temporal DCT Complexity: {temporal_dct_complexity:.2f}")
    print(f"Histogram Complexity: {histogram_complexity:.2f}")
    print(f"Edge Detection Complexity: {edge_detection_complexity:.2f}")

    # Calculate average complexity across all methods
    average_complexity = (advanced_motion_complexity + dct_complexity + temporal_dct_complexity + 
                          histogram_complexity + edge_detection_complexity) / 5
    return average_complexity

def run_ffmpeg(input_video, crf, output_video, vmaf_model_path):
    # Construct FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', 'medium',
        '-lavfi', f'psnr="stats_file=psnr.log"',
        '-lavfi', f'ssim="stats_file=ssim.log"',
        '-lavfi', f'libvmaf=model_path={vmaf_model_path}:log_path=vmaf.log',
        output_video
    ]
    
    # Run FFmpeg command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    return stdout, stderr

def extract_metrics_from_logs(psnr_log, ssim_log, vmaf_log, video_file, crf, bitrate, resolution, frame_rate):
    # Initialize metrics
    psnr = None
    ssim = None
    vmaf = None

    # Extract PSNR value from the log file
    with open(psnr_log) as f:
        for line in f:
            if "average:" in line:
                psnr = float(line.split("average:")[-1].strip())

    # Extract SSIM value from the log file
    with open(ssim_log) as f:
        for line in f:
            if "All:" in line:
                ssim = float(line.split("All:")[-1].strip())

    # Extract VMAF value from the log file
    with open(vmaf_log) as f:
        for line in f:
            if "VMAF score" in line:
                vmaf = float(line.split("VMAF score:")[-1].strip())

    # Calculate scene complexity
    scene_complexity = calculate_average_scene_complexity(video_file)

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

    # Append to the CSV file
    df.to_csv(csv_file, mode='a', index=False, header=not file_exists)
    
def get_video_info(video_path):
    # Use FFmpeg to retrieve video properties
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-hide_banner'
    ]

    # Run FFmpeg command to get video info
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Parse bitrate, resolution, and frame rate from stderr
    bitrate, resolution, frame_rate = None, None, None
    stderr_lines = stderr.decode().split('\n')

    for line in stderr_lines:
        # Get bitrate
        if "bitrate:" in line:
            bitrate = int(re.search(r'(\d+)', line.split('bitrate:')[-1]).group(0))
        # Get resolution
        if "Video:" in line:
            resolution_match = re.search(r'(\d+x\d+)', line)
            if resolution_match:
                resolution = resolution_match.group(0)
        # Get frame rate
        if "fps" in line:
            frame_rate_match = re.search(r'(\d+(\.\d+)?) fps', line)
            if frame_rate_match:
                frame_rate = float(frame_rate_match.group(1))

    return bitrate, resolution, frame_rate

# Call extract_metrics_from_logs after running FFmpeg
def process_video_and_extract_metrics(input_video, crf, output_video, vmaf_model_path):
    # Run FFmpeg and generate logs
    run_ffmpeg(input_video, crf, output_video, vmaf_model_path)
    
    # Extract bitrate, resolution, and frame rate from video
    bitrate, resolution, frame_rate = get_video_info(input_video)

    # Extract metrics from the generated logs (PSNR, SSIM, VMAF)
    metrics = extract_metrics_from_logs(
        psnr_log='psnr.log',
        ssim_log='ssim.log',
        vmaf_log='vmaf.log',
        video_file=input_video,
        crf=crf,
        bitrate=bitrate,
        resolution=resolution,
        frame_rate=frame_rate
    )

    print(metrics)

    # Update the CSV file with the new metrics
    update_csv(metrics, csv_file='video_quality_data.csv')
    
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
    # Parse the command-line arguments
    args = parse_arguments()

    # Call the main function with the provided arguments
    process_video_and_extract_metrics(
        input_video=args.input_video,
        crf=args.crf,
        output_video=args.output_video,
        vmaf_model_path=args.vmaf_model_path
    )
    
    print(f"Processing completed for video: {args.input_video}")