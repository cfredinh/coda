#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pdb
import argparse
from glob import glob
from PIL import Image

import multiprocessing
from joblib import Parallel, delayed


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--target_dir', type=str, required=True, 
                        help= 'Give the path of the images.')
    parser.add_argument('--n_channels', type=int, required=True, default=3, 
                        help= 'Give the number of the channels that each image has.')    
    parser.add_argument('--image_size', type=int, required=False, default=256, 
                        help= 'Size of the updated images.')
    parser.add_argument('--image_types', type=list, required=False, default=["png", "jpg", "JPG"], 
                        help= 'Type of the original images.')   
    parser.add_argument('--target_image_type', type=str, required=False, default="png", 
                        help= 'Type of the target images.')       
    parser.add_argument('--new_dir', type=str, required=False, default="", 
                        help= 'Path of the new dir')        
    return parser.parse_args()

    
    
def check_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except:
        pass
        
def pil_loader(img_path, n_channels):
    if img_path.endswith(('.tif', '.tiff')):
        return Image.fromarray(io.imread(img_path))
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 3:
            return img.convert('RGB')
        elif n_channels == 1:
            return img.convert('L')
        elif n_channels ==4:
            return img.convert('RGBA')
        else:
            raise NotImplementedError("PIL only supports 1,3 and 4 channel inputs. Use cv2 instead")        
        
def resize(org_path, target_dir, new_dir, size, n_channels, target_image_type):
    rel_path = org_path.split(target_dir)[-1]
    rel_path = rel_path[1:] if rel_path.startswith("/") else rel_path
    new_path = os.path.join(new_dir, rel_path)
    new_path = '.'.join(new_path.split('.')[:-1]) + f'.{target_image_type}'
    img = pil_loader(org_path, n_channels)
    img = img.resize(size, resample=Image.Resampling.LANCZOS)
    check_dir(os.path.dirname(new_path))
    img.save(new_path)
    img.close()
    

def main(args):
    target_dir = os.path.abspath(args.target_dir)
    new_dir = args.new_dir
    n_channels = args.n_channels
    
    if not new_dir:
        new_dir = f"{target_dir}_{args.image_size}"
    check_dir(new_dir)
    image_types = args.image_types
    target_image_type = args.target_image_type
    img_size = (args.image_size, args.image_size)
    all_img_paths = [] 
    for image_type in image_types:
        all_img_paths += glob(os.path.join(target_dir, "**", f"*.{image_type}"), recursive=True)

    n_jobs = 1 if multiprocessing.cpu_count() == 0 else -3

    print('Resizing starts . . .')
    Parallel(n_jobs=n_jobs, verbose=1)(delayed(resize)(
        all_img_paths[i], target_dir, new_dir, img_size, n_channels, target_image_type)
        for i in range(len(all_img_paths)))  
#     for i in range(len(all_img_paths)):
#         resize(all_img_paths[i], target_dir, new_dir, img_size, n_channels, target_image_type)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)