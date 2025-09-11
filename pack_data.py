import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import json
import multiprocessing as mp
import logging
import time
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def pack_single_file(args):
    """
    Helper function to pack a single file (needed for multiprocessing)
    
    Args:
        args (tuple): Tuple containing (file_path, output_dir)
    """
    file_paths, idx, output_dir = args
    return pack_torch_file(file_paths, idx, output_dir)

def pack_torch_file(file_paths, idx, output_dir):
    """
    Process a .torch file and save images and poses
    
    Args:
        file_paths (str): Files that should be saved in a .torch file
        output_dir (str): Base directory to save outputs
    """
    data = []

    try:
        os.makedirs(output_dir, exist_ok=True)
        for file_path in file_paths:
            cur_scene = {}
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            cur_scene['key'] = metadata['scene_name']
            cur_scene['images'] = []
            cur_scene['cameras'] = []

            for frame_idx, frame in enumerate(metadata['frames']):
                try:
                    img_array = cv2.imread(frame['image_path'])
                    h, w = img_array.shape[:2]
                    success, img_array = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    assert success
                    img_data = torch.from_numpy(img_array)
                    cur_scene['images'].append(img_data)

                    w2c = torch.from_numpy(np.array(frame['w2c'][:3], dtype=np.float32).reshape(12))
                    fx, fy, cx, cy = frame['fxfycxcy']
                    pose_data = torch.cat((
                        torch.tensor([
                            fx / w,
                            fy / h,
                            cx / w,
                            cy / h,
                            0.,
                            0.,
                        ]).float(),
                        w2c,
                    ))
                    cur_scene['cameras'].append(pose_data)
                except Exception as e:
                    logging.error(f"Error packing frame {frame_idx} in {file_path}: {str(e)}")
                    continue
        
            cur_scene['cameras'] = torch.stack(cur_scene['cameras'])
            data.append(cur_scene)

        torch.save(data, os.path.join(output_dir, f'{idx:06}.torch'))
        return True, file_paths
    except Exception as e:
        logging.error(f"Error packing {file_paths}: {str(e)}")
        return False, file_paths

def pack_directory(input_dir, output_dir, num_processes=None, chunk_size=1, num_scenes_per_torch_chunk=16):
    """
    Process all .torch files in a directory using multiprocessing
    
    Args:
        input_dir (str): Directory containing full_list.txt, metadata, and images
        output_dir (str): Directory containing .torch files
        num_processes (int, optional): Number of processes to use. Defaults to CPU count - 1
        chunk_size (int, optional): Number of files to pack per worker at once. Defaults to 1
    """
    with open(os.path.join(input_dir, 'full_list.txt'), 'r') as f:
        full_list = [x.rstrip() for x in f.readlines()]

    total_files = len(full_list)

    if full_list and not os.path.isabs(full_list[0]):
        logging.warning(f"File paths in full_list.txt are relative")
    
    # Set up multiprocessing
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    # Prepare arguments for multiprocessing
    args = [
        (full_list[i:i+num_scenes_per_torch_chunk], idx, output_dir)
        for idx, i in enumerate(range(0, total_files, num_scenes_per_torch_chunk))
    ]

    logging.info(f"Found {total_files} scenes to pack in {input_dir} (result in {len(args)} .torch files)")
    
    # Process files in parallel with progress bar
    start_time = time.time()
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(pack_single_file, args, chunksize=chunk_size),
            total=len(args),
            desc=f"Packing files with {num_processes} processes"
        ))
    
    # Log results
    successful = sum(1 for success, _ in results if success)
    failed = [(success, paths) for success, paths in results if not success]
    
    elapsed_time = time.time() - start_time
    logging.info(f"Packing completed in {elapsed_time:.2f} seconds")
    logging.info(f"Successfully packed {successful}/{len(args)} files")
    
    if failed:
        logging.warning(f"Failed to pack {sum([len(paths) for _, paths in failed])} files:")
        for _, paths in failed:
            for path in paths:
                logging.warning(f"  - {path}")

def generate_full_list(base_path, output_dir):
    # find all .json files in the base_path and generate a full list saving their absolute paths and store it in a txt file in the output_dir
    json_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.json')]
    json_files = [os.path.abspath(f) for f in json_files]
    json_files.sort()
    with open(os.path.join(output_dir, 'full_list.txt'), 'w') as f:
        for file in json_files:
            f.write(file + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=10, help='chunksize used in multiprocessing.Pool.imap')
    parser.add_argument("--num_scenes_per_torch_chunk", '-ns', type=int, default=16, help='number of scenes for each .torch file')
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default='/share/phoenix/nfs06/S9/hj453/DATA/re10k_raw/')
    parser.add_argument("--base_path", type=str, default='/share/phoenix/nfs06/S9/hj453/DATA/re10k/')
    
    args = parser.parse_args()

    for split in ['train', 'test']:
        input_dir = os.path.join(args.base_path, split)
        output_dir = os.path.join(args.output_dir, split)
        logging.info(f"Starting {split} data packing...")
        pack_directory(input_dir, output_dir, chunk_size=args.chunk_size, num_processes=args.num_processes, num_scenes_per_torch_chunk=args.num_scenes_per_torch_chunk)  
        logging.info("Packing completed!")
