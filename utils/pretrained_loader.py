import os
import torch
from pprint import pprint

from collections import OrderedDict

def state_dict_from_path(path):
    """
    Returns the state dict from a checkpoint path
    """
    path = os.path.abspath(path)
    mname = os.path.basename(path)
    if os.path.isfile(path):
        print("Loading weights from \"{}\"".format(mname))
        return torch.load(path)['state_dict']
    else:
        dirname = os.path.dirname(path)
        raise FileNotFoundError(
            "Model \"{}\" is not present in \"{}\"".format(mname, dirname))

def load_from_pretrained(model, path, strict=False, load_backbone = True):
    """
    Loads wights to the input model from a pretrained model path
    """
    pretrained_state = state_dict_from_path(path)
    

    if load_backbone:
        pretrained_state = OrderedDict([(k.split("backbone.")[1], v) 
                                                    for k,v in pretrained_state.items() 
                                                    if 'backbone' in k])  

        dif_keys = model.backbone.load_state_dict(pretrained_state, strict=strict)
        dif_keys = set([" : ".join(key.split(".")[:2]) for key in dif_keys.unexpected_keys])

    else:
        dif_keys = model.load_state_dict(pretrained_state, strict=strict)
        dif_keys = set([" : ".join(key.split(".")[:2]) for key in dif_keys.unexpected_keys])

    if dif_keys:
        print("Unmatched pretrained modules")
        pprint(dif_)



def load_from_pretrained_special(model, path, strict=False, load_backbone = True):
    """
    Loads wights to the input model from a pretrained model path
    """
    pretrained_state = state_dict_from_path(path)
    

    pretrained_state = OrderedDict([(k.split("teacher_encoder.")[1], v) 
                                            for k,v in pretrained_state.items() 
                                            if 'teacher_encoder' in k])  
    dif_keys = model.load_state_dict(pretrained_state, strict=strict)
    dif_keys = set([" : ".join(key.split(".")[:2]) for key in dif_keys.unexpected_keys])
        
    if dif_keys:

        print("Unmatched pretrained modules")
        pprint(dif_keys)
