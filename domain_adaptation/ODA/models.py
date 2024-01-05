from defaults.bases import *
from defaults.models import Identity
from torch.cuda.amp import autocast

__all__ = ['Ymodel']

class Ymodel(BaseModel):
    def __init__(self, model_params, foundation_model, main_model):
        super().__init__()
        self.attr_from_dict(model_params)
        self.foundation_params = self.ODA.foundation_params
        self.frozen_main = self.freeze_backbone
        self.frozen_foundation = self.foundation_params.freeze_backbone 
        self.embed_dim = main_model.embed_dim
        
        self.main_model = main_model
        self.foundation_model = foundation_model
        self.linear_projector = Identity()

        self.linear_projector = nn.Linear(self.foundation_model.embed_dim, self.main_model.embed_dim)
        
        # Initialize weights
        self.load_weights()
        
        # freezing foundation (and potentialy main)
        self.freeze_check()
        
        # send the wrapped model to the original model's GPU ID
        self.to(self.device_id)
                

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            self.freeze_check()
            with torch.set_grad_enabled(not self.frozen_foundation):
                x = self.foundation_model(x, return_all=True) 
            x = self.linear_projector(x)
            x = self.main_model(x, return_embedding=return_embedding, use_patchifier=False)
            
            return x

    def freeze_check(self):
        self.freeze_foundation()
        self.freeze_main()
        
    def freeze_foundation(self):
        if self.frozen_foundation:
            self.foundation_model.eval()            
            self.freeze_submodel(self.foundation_model)
            
    def freeze_main(self):
        if self.frozen_main:
            self.main_model.eval() 
            self.freeze_submodel(self.main_model)
            self.linear_projector.eval()            
            self.freeze_submodel(self.linear_projector)
            
    def load_weights(self):
        foundation_TL_params = self.foundation_params.transfer_learning_params
        main_TL_params       = self.ODA.main_model_params.transfer_learning_params
        
        # Transfer foundation model
        if foundation_TL_params.use_pretrained:
            # Getting args and paths for pretrained state
            pretrained_path       = foundation_TL_params.pretrained_path
            pretrained_method     = foundation_TL_params.pretrained_method
            pretrained_model_name = foundation_TL_params.pretrained_model_name
            if not pretrained_path:
                pretrained_path = os.path.join(self.save_dir, "checkpoints")
            pretrained_path     = os.path.join(pretrained_path, pretrained_model_name)
            pretrained_state = state_dict_from_path(pretrained_path)
            
            # Clean up the state dict based on the pretrained method
            if pretrained_method in ['dino', 'DINO']:
                pretrained_state = OrderedDict([(k.split("teacher_encoder.")[1], v) 
                                                for k,v in pretrained_state.items() 
                                                if 'teacher_encoder' in k])         
            elif pretrained_method in ['oda', 'ODA']:
                pretrained_state = OrderedDict([(k.split("foundation_model.")[1], v) 
                                                for k,v in pretrained_state.items() 
                                                if 'foundation_model' in k]) 
            elif pretrained_method in ['supervised']:
                pass                
            elif pretrained_method in ['byol', 'BYOL']:
                raise NotImplemented
            else:
                raise ValueError(f"ODA is does not support {pretrained_method} as an SSL type.")
            
            # Load the cleaned state dict
            dif_keys = self.foundation_model.load_state_dict(pretrained_state, strict=False)
            print(dif_keys)
            assert not dif_keys.missing_keys, "\tThe pretrained state did not much the foundation model"
            print_ddp("\033[92m\tPretrained weights are loaded to the foundation model\033[0m")
            
        
        # Transfer main model
        if main_TL_params.use_pretrained:
            # Getting args and paths for pretrained state
            pretrained_path       = main_TL_params.pretrained_path
            pretrained_method     = main_TL_params.pretrained_method
            pretrained_model_name = main_TL_params.pretrained_model_name
            if not pretrained_path:
                pretrained_path = os.path.join(self.save_dir, "checkpoints")
            pretrained_path = os.path.join(pretrained_path, pretrained_model_name)
            pretrained_state = state_dict_from_path(pretrained_path) 
            
            
            # Adapt the state dict based on the pretrained method
            if pretrained_method in ['oda', 'ODA']:
                # first load the linear state
                linear_state = OrderedDict([(k.split("linear_projector.")[1], v) 
                                                for k,v in pretrained_state.items() 
                                                if 'linear_projector' in k])
                self.linear_projector.load_state_dict(linear_state, strict=True)
                
                # load the rest of the main model
                pretrained_state = OrderedDict([(k.split("main_model.")[1], v) 
                                                for k,v in pretrained_state.items() 
                                                if 'main_model' in k])  
            elif pretrained_method in ['dino', 'DINO']:
                pretrained_state = OrderedDict([(k.split("teacher_encoder.")[1], v) 
                                                for k,v in pretrained_state.items() 
                                                if 'teacher_encoder' in k])           
            elif pretrained_method in ['supervised']:
                pass
            else:
                raise ValueError(f"ODA is does not support {pretrained_method} as an SSL type.")
            
            # Load the cleaned state dict
            dif_keys = self.main_model.load_state_dict(pretrained_state, strict=False)
            if not self.frozen_main:
                keycheck =  set(dif_keys.missing_keys) - set(['fc.bias', 'fc.weight'])
            else:
                keycheck =  dif_keys.missing_keys
            assert not keycheck, "\tThe pretrained state did not much the main model"
            print_ddp("\033[92m\tPretrained weights are loaded to the main model\033[0m")             
            

            