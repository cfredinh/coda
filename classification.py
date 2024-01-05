#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import wandb
import argparse

import torch
from utils.system_def import *
from utils.launch import dist, launch, synchronize

from defaults import *
from self_supervised.BYOL import *
from self_supervised.DINO import *

from domain_adaptation.ODA import *


global debug


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="./params.json",
                        help='Give the path of the json file which contains the training parameters')
    parser.add_argument('--checkpoint', type=str, required=False, help='Give a valid checkpoint name')
    parser.add_argument('--test', action='store_true', default=False, help='Flag for testing')
    
    parser.add_argument('--debug', action='store_true', default=False, help='Flag for turning on the debug_mode')
    parser.add_argument('--data_location', type=str, required=False, help='Update the datapath')
    parser.add_argument('--dist_url', type=str, default='', required=False,
                        help='URL of master node, for use with SLURM')
    parser.add_argument('--port', type=int, required=False, default=45124, 
                        help='Explicit port selection, for use with SLURM')
    parser.add_argument('--gpu', type=str, required=False, help='The GPU to be used for this run')
    parser.add_argument('--model_name', type=str, required=False, help='Used to manipulate the model_name defined in the param file for this run')
    parser.add_argument('--save_dir', type=str, required=False, help='Change the "save_dir" in param file')
    
    parser.add_argument('--backbone_type', type=str, help='Change backbone')
    parser.add_argument('--batch_size', type=int, help='Change the batch size of all data loaders')
    parser.add_argument('--patch_size', type=int, help='Change model patch size')
    parser.add_argument('--val_every', type=float, help='How many epochs between each validation')
    parser.add_argument('--use_mixed_precision', type=int, default=-1, help='Using this arg with value 1 (0) results in explicity turning on (off) the mixed precision. By default it follows the param file.')
    
    # semi/self args
    parser.add_argument('--byol',    action='store_true', default=False, help='Flag for training with BYOL')
    parser.add_argument('--simsiam', action='store_true', default=False, help='Flag for training with SimSiam')
    parser.add_argument('--dino',    action='store_true', default=False, help='Flag for training with DINO')
    parser.add_argument('--mae',     action='store_true', default=False, help='Flag for training with MAE')    
    
    # domain adaptation args
    parser.add_argument('--save_oda_model',            action='store_true', default=False, 
                        help='Flag for loading and saving the trained models (as defined in json)')    
    parser.add_argument('--oda',            action='store_true', default=False, 
                        help='Flag for training and testing using Foundation models')    
    parser.add_argument('--ssl_oda',        action='store_true', default=False, 
                        help='Flag for changing the SSL modules to use the test set for ODA')
    parser.add_argument('--test_all', action='store_true', default=False, 
                        help='Flag for testing on the full dataset (not only the test subest)') 
    
    

    return parser.parse_args()


def update_params_from_args(params, args):
    if args.gpu:
        prev_gpu = params.system_params.which_GPUs
        params.system_params.which_GPUs = args.gpu  # change the value in-place
        print('Changed GPU for this run from {} to \033[1m{}\033[0m'.format(prev_gpu, args.gpu))

    if args.model_name:
        prev_model_name = params.training_params.model_name
        params.training_params.model_name = args.model_name
        print('Changed model_name for this run from {} to \033[1m{}\033[0m'.format(prev_model_name, args.model_name))

    if args.data_location:
        params['dataset_params']['data_location'] = args.data_location
        print('Changed data_location to: "\033[1m{}\033[0m"'.format(args.data_location))

    if args.save_dir:
        params['training_params']['save_dir'] = args.save_dir
        print('Changed save_dir to: "\033[1m{}\033[0m"'.format(args.save_dir))

    if args.backbone_type:
        # update_name = True if args.model_name is None else False
        params['model_params']['backbone_type'] = args.backbone_type
        print('Changed backbone_type to: \033[1m{}\033[0m'.format(args.backbone_type))

    if args.batch_size:
        for loader in ['trainloader', 'valloader', 'testloader']:
            params['dataloader_params'][loader]['batch_size'] = args.batch_size
            print('Changed \033[1m{}\033[0m batch_size to: \033[1m{}\033[0m'.format(loader, args.batch_size))

    if args.patch_size:
        params['model_params']['transformers_params']['patch_size'] = args.patch_size
        print('Changed patch_size to: \033[1m{}\033[0m'.format(args.patch_size))

    if args.val_every is not None:
        params['training_params']['val_every'] = args.val_every
        print('CHanged val_every to: \033[1m{}\033[0m'.format(args.val_every))

    if args.use_mixed_precision != -1:  # if -1, it won't change the param file
        assert args.use_mixed_precision == 0 or args.use_mixed_precision == 1, 'Argument --use_mixed_precision shold be either 1 or 0'
        if_mixed = bool(args.use_mixed_precision)  # this step is not crucial but is good for consistency
        params['training_params']['use_mixed_precision'] = if_mixed
        print('Explicitly set use_mixed_precision to: \033[1m{}\033[0m'.format(if_mixed))
        
        
    if args.save_oda_model:
        args.oda = True
        args.test = True
        print("Setting mode to Online DA (-oda)")
        print("Test flag: True")     


def main(parameters, args):
    # check self-supervised method
    assert not args.byol * args.simsiam, "BYOL or SimSiam can be on but not both"
    use_momentum = True if args.byol else False 
    
    # define system
    define_system_params(parameters.system_params)
    
    # Instantiate wrapper with all its definitions   
    if args.byol or args.simsiam or args.dino:
        if args.dino:
            if args.ssl_oda:
                wrapper = TestTimeSSLWrapper(parameters, ssl_type='dino')
            else:
                wrapper = DINOWrapper(parameters)
        else:
            wrapper = BYOLWrapper(parameters, use_momentum=use_momentum)
    elif args.mae:
        if args.ssl_oda:
            wrapper = TestTimeSSLWrapper(parameters, ssl_type='mae')
        else:
            wrapper = MAEWrapper(parameters)             
    elif args.oda:
        wrapper = TestTimeWrapper(parameters) 
    else:
        wrapper = DefaultWrapper(parameters, test_all=args.test_all)
    wrapper.instantiate()

    # initialize logger
    if wrapper.is_rank0:
        log_params = wrapper.parameters.log_params    
        training_params = wrapper.parameters.training_params
        if wrapper.log_params['run_name'] == "DEFINED_BY_MODEL_NAME":
            log_params['run_name'] = training_params.model_name  
        if args.debug:
            os.environ['WANDB_MODE'] = 'dryrun'
        if not args.test:
            if parameters.training_params.use_tensorboard:
                print("Using TensorBoard logging")
                summary_writer = SummaryWriter()
            else:
                print("Using WANDB logging")
                wandb.init(project=log_params.project_name, 
                           name=log_params.run_name, 
                           config=wrapper.parameters,
                           resume=True if training_params.restore_session else False)
    
    # define trainer 
    if args.byol or args.simsiam or args.dino:
        if args.dino:
            trainer = DINOTrainer(wrapper)
        else:
            trainer = BYOLTrainer(wrapper, use_momentum)       
    elif args.oda:
        trainer = OnlineDA(wrapper) 
    else:
        trainer = Trainer(wrapper)
        
    if parameters.training_params.use_tensorboard:
        trainer.summary_writer = summary_writer
        
    if args.save_oda_model:
        trainer.save_session(verbose=True)
        return
        
    if args.test:
        trainer.test(test_all=args.test_all)               
    else:
        trainer.train()
        if wrapper.is_supervised:
            trainer.test(test_all=args.test_all)
        
    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = edict(load_params(args))
    update_params_from_args(parameters, args)

    try:
        launch(main, (parameters, args))
    except Exception as e:       
        if dist.is_initialized():
            dist.destroy_process_group()            
        raise e
    finally:
        if dist.is_initialized():
            synchronize()         
            dist.destroy_process_group()            
