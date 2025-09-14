import argparse
import json
import os
import sys
import multiprocessing as mp
from tqdm import tqdm

def func(scene_path):
    data_json = json.load(open(scene_path, 'r'))
    frames = data_json["frames"]
    valid_cnt = sum([int(os.path.exists(frame['image_path'])) for frame in frames])
    descs = []
    if valid_cnt != len(frames):
        desc = f'invalid image_path: {scene_path} ({valid_cnt}/{len(frames)})\n'
        sys.stderr.write(desc)
        descs.append(desc)
    if len(frames) != 24:
        desc = f'invalid frame length: {scene_path} ({len(frames)}/24)\n'
        sys.stderr.write(desc)
        descs.append(desc)
    return descs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    with open(args.dataset, 'r') as f:
        all_scene_paths = f.read().splitlines()
    all_scene_paths = [path.strip() for path in all_scene_paths if path.strip()]

    descs = []

    with mp.Pool(processes=32) as pool:
        descs = sum(list(tqdm(pool.imap(func, all_scene_paths, chunksize=128), total=len(all_scene_paths))), [])

    for desc in descs:
        sys.stderr.write(desc)
