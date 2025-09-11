#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from functools import partial
import torchvision.transforms as T # Import torchvision transforms for resizing
from multiprocessing import Pool, cpu_count # Import multiprocessing modules

def calculate_intrinsics_from_fovx(fov_x_rad, image_width, image_height):
    """
    Calculates camera intrinsic parameters (fx, fy, cx, cy) from horizontal FOV and image dimensions.
    """
    if fov_x_rad is None or fov_x_rad <= 0 or image_width <= 0 or image_height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    fx = 0.5 * image_width / math.tan(fov_x_rad / 2.0)
    fy = fx # Assuming square pixels
    cx = image_width / 2.0
    cy = image_height / 2.0
    return [fx, fy, cx, cy]

def process_trellis_scene(scene_dir, lvsm_output_dir):
    """
    Processes a single Trellis scene directory:
    - Reads transforms.json.
    - Resizes and copies images to the LVSM images directory.
    - Calculates intrinsics based on the new image size.
    - Generates and saves a scene-specific JSON metadata file for LVSM.
    """
    scene_id = os.path.basename(scene_dir)
    transformer_json_path = os.path.join(scene_dir, "transforms.json")

    if not os.path.exists(transformer_json_path):
        print(f"Warning: 'transforms.json' not found in {scene_dir}. Skipping scene.")
        return False # Return False to indicate failure/skip

    try:
        with open(transformer_json_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {transformer_json_path}. Skipping scene.")
        return False

    # Construct output directories using the absolute LVSM root path
    images_output_dir = os.path.join(lvsm_output_dir, "images", scene_id)
    metadata_output_dir = os.path.join(lvsm_output_dir, "metadata")
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(metadata_output_dir, exist_ok=True)

    lvsm_metadata = {"scene_name": scene_id, "frames": []}

    # Define the image resize transform to 256x256
    # antialias=True provides better quality but requires Pillow 9.1.0+
    resize_transform = T.Resize((512, 512), antialias=True) 

    for frame in metadata.get("frames", []):
        relative_img_path = frame.get("file_path")
        if relative_img_path is None:
            print(f"Warning: Frame missing 'file_path' in scene {scene_id}. Skipping frame.")
            continue
        
        src_img_basename = os.path.basename(relative_img_path)
        src_img_path = os.path.join(scene_dir, src_img_basename)

        if not os.path.exists(src_img_path):
            print(f"Warning: Image file not found at {src_img_path}. Skipping frame.")
            continue
        
        try:
            # Open image, convert to RGBA, and resize
            with Image.open(src_img_path).convert("RGBA") as img:
                assert img.size == (1024, 1024)
                white_background = Image.new("RGB", img.size, (127, 127, 127))
                white_background.paste(img, (0, 0), img)
                img_resized = resize_transform(white_background).crop((128, 128, 384, 384))
                # Get the new dimensions after resizing for intrinsic calculation
                image_width, image_height = img_resized.size 
                
                # Save the resized image
                dst_img_name = f"{len(lvsm_metadata['frames']):05d}.png"
                dst_img_path = os.path.join(images_output_dir, dst_img_name)
                img_resized.save(dst_img_path) # Save the resized image directly
        except Exception as e:
            print(f"Error processing image {src_img_path}: {e}. Skipping frame.")
            continue

        # 从 transforms.json 读取的是 c2w (camera-to-world) 矩阵
        c2w_matrix = frame.get("transform_matrix", None)
        if c2w_matrix is None:
            print(f"Warning: 'transform_matrix' not found for a frame in scene {scene_id}. Skipping frame.")
            continue

        # 将 c2w 矩阵转换为 numpy 数组
        c2w_np = np.array(c2w_matrix)

        # 检查矩阵是否为可逆的
        if np.linalg.det(c2w_np) == 0:
            print(f"Warning: Non-invertible c2w matrix found for a frame in scene {scene_id}. Skipping frame.")
            continue

        # 通过对 c2w 矩阵求逆来得到 w2c 矩阵
        w2c_np = np.linalg.inv(c2w_np)

        # 将 numpy 数组转换回列表格式以方便 JSON 序列化
        w2c_matrix = w2c_np.tolist()

        fov_x = frame.get("camera_angle_x")
        # Use the resized image dimensions for intrinsic calculation
        fxfycxcy = calculate_intrinsics_from_fovx(fov_x, 512, 512)[:2] + calculate_intrinsics_from_fovx(fov_x, 256, 256)[2:]
        
        if fxfycxcy == [0.0, 0.0, 0.0, 0.0]:
            print(f"Warning: Invalid 'camera_angle_x' or image dimensions for a frame in scene {scene_id}. Using default (zero) intrinsics.")

        frame_data = {
            # Image path is now absolute
            "image_path": os.path.abspath(dst_img_path), 
            "fxfycxcy": fxfycxcy,
            "w2c": w2c_matrix
        }
        lvsm_metadata["frames"].append(frame_data)

    if not lvsm_metadata["frames"]:
        print(f"Warning: No valid frames were processed for scene {scene_id}. Skipping metadata saving.")
        return False

    output_metadata_path = os.path.join(metadata_output_dir, f"{scene_id}.json")
    with open(output_metadata_path, 'w') as f:
        json.dump(lvsm_metadata, f, indent=4)
    # print(f"Processed scene {scene_id} with {len(lvsm_metadata['frames'])} valid frames.")
    return True # Return True to indicate successful processing

def generate_full_list(metadata_dir, output_file):
    """
    Generates a full_list.txt file containing absolute paths to all scene metadata JSON files.
    """
    json_files = [
        os.path.abspath(os.path.join(metadata_dir, f))
        for f in sorted(os.listdir(metadata_dir))
        if f.endswith(".json")
    ]
    
    with open(output_file, "w") as f:
        f.write("\n".join(json_files) + "\n")
    print(f"Generated full_list.txt at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Trellis dataset to LVSM RealEstate10K format.")
    parser.add_argument("--trellis_metadata", required=True)
    # This parameter is kept for compatibility but the actual output target is hardcoded in process_trellis_scene.
    parser.add_argument("--lvsm_output_dir", required=True, 
                        help="Path to the output directory for LVSM formatted data (will create images/ and metadata/ subdirs).")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel processes to use for conversion. Default is CPU cores - 1.")
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=10)
    args = parser.parse_args()

    # Discover all scene directories within the provided Trellis data directory
    with open(args.trellis_metadata, 'r') as f:
        scene_dirs = json.load(f)
    split = int(len(scene_dirs) * args.test_split)

    for scenes, split_name in [(scene_dirs[:split], 'test'), (scene_dirs[split:], 'train')]:
        print(f"Found {len(scenes)} scenes to process for {split_name}.")

        # Use multiprocessing Pool to process scenes in parallel
        # The `if __name__ == "__main__":` block is crucial for multiprocessing on Windows.
        with Pool(processes=args.num_workers) as pool:
            # imap_unordered is used for efficient, unordered processing with a progress bar
            for _ in tqdm(pool.imap_unordered(partial(process_trellis_scene, lvsm_output_dir=os.path.join(args.lvsm_output_dir, split_name)), scenes, chunksize=args.chunk_size), 
                        total=len(scenes), desc=f"Processing Trellis scenes for {split_name}", 
                        unit="scene"):
                pass # Iterate results to update the tqdm progress bar

        # After all scenes are processed, generate the full_list.txt
        # The final LVSM output root path is hardcoded as per requirements
        lvsm_metadata_dir = os.path.join(args.lvsm_output_dir, split_name, "metadata")
        os.makedirs(lvsm_metadata_dir, exist_ok=True)
        generate_full_list(lvsm_metadata_dir, os.path.join(args.lvsm_output_dir, split_name, "full_list.txt"))

        print(f"\nConversion complete. LVSM formatted data saved to: {args.lvsm_output_dir}/{split_name}")

