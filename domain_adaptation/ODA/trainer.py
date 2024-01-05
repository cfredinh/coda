import wandb
from defaults.trainer import *
from self_supervised.BYOL import BYOLTrainer
from self_supervised.DINO import DINOTrainer
from .models import Ymodel

#  __all__ = ['OnlineDA']

class OnlineDA(Trainer):
    def __init__(self, wraped_defs, single_shot=False, mae_iters=5):
        super().__init__(wraped_defs)
        if single_shot:
            self.mae_iters = mae_iters
            self.org_mae_state       = model_to_CPU_state(self.model.mae)
            self.org_Ymodel_state    = model_to_CPU_state(self.model.y_model)        
            self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer) 
            
            self.oda_train_transform = wraped_defs.dataloaders.trainloader.dataset.transform
            
            # Create the new Compose sequence
            new_transform_list = [t for i, t in enumerate(self.oda_train_transform.transforms) if i not in [0,1,4,5,6,7]]
            self.oda_train_transform = transforms.Compose(new_transform_list)
            self.test = self.test_single_shot
    
    def reset_Ymodel(self, model=None):
        if model is None: model = self.model.y_model
        model.load_state_dict(self.org_Ymodel_state)
        model.to(self.device_id)
        