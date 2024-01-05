#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')
from helpfuns import *
from defaults import *
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="../params.json",
                        help= 'Give the path of the json file which contains the training parameters')    
    parser.add_argument('--image_size', type=int, required=False, default=256, 
                        help= 'Size of the updated images')
    parser.add_argument('--num_workers', type=int, required=False, default=18, 
                        help= 'Number of workers')    
    parser.add_argument('--for_ssl', action='store_true', default=False, 
                        help= 'Flag for using all the data meant for SSL')    
    return parser.parse_args()
    
def compute_stats(dataset=None, dataloader=None, img_size=None):
    from tqdm.notebook import tqdm
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import Compose, Resize, ToTensor 
    if dataset==None and dataloader==None:
        raise ValueError("Please give as argumets a dataloader or a dataset")
        
    dataloader_params = {'batch_size': 100,
                         'num_workers': 12,
                         'prefetch_factor': 1,
                         'shuffle': False,
                         'pin_memory': False,
                         'drop_last': False,
                         'persistent_workers': False}        
    if dataloader is not None:
        print("Using given dataloader to compute stats")
        pass
    elif dataset is not None:
        print("Creating new dataloader with the only augmentations being resizing")
        if img_size is not None:
            dataset.transform = Compose([Resize(img_size), ToTensor()])      
        else:
            dataset.transform = Compose([ToTensor()])                  
        dataloader = DataLoader(dataset, **dataloader_params)
    else:
        raise ValueError("Input args not understood or ill-defiined")

    channels = dataloader.dataset[0][0].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x, _ in tqdm(dataloader):
        x_tot += x.mean([0,2,3]).cpu().numpy()
        x2_tot += (x**2).mean([0,2,3]).cpu().numpy()

    channel_avr = x_tot/len(dataloader)
    channel_std = np.sqrt(x2_tot/len(dataloader) - channel_avr**2)
    return channel_avr,channel_std    
    
def main(parameters, args):
    img_size = (args.image_size, args.image_size)
    dataset_params = parameters.dataset_params
    dataset_name = dataset_params.dataset
    dataset_params.ssl_mode = True if args.for_ssl else False
    
    dataset = getattr(sys.modules['defaults.datasets'], dataset_name) 
    dataset = dataset(dataset_params=dataset_params, mode='train')

    print(f"{'*'*10}\nCalculating stats for {dataset_name}")
    channel_avr, channel_std = compute_stats(dataset=dataset, img_size=img_size)
    channel_avr, channel_std = channel_avr.round(3), channel_std.round(3)
    print(f"{'*'*10}\n {dataset_name}: mean: {channel_avr} \n {dataset_name}: std: {channel_std}\n")
        
    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = edict(load_json(args.params_path))
    main(parameters, args)    
    