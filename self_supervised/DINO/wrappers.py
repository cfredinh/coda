from defaults import *
from utils.helpfuns import *
from .models import DINO
from self_supervised.BYOL.wrappers import BYOLWrapper

__all__ = ['DINOWrapper']

class DINOWrapper(BYOLWrapper):
    def __init__(self, parameters, aug_strat_path=None, test_all=False):
        super().__init__(parameters, aug_strat_path=None, test_all=test_all)
        
        model_type = parameters.model_params.backbone_type
        if hasattr(transformers.swin, model_type):
            self_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
            aug_strat_path = os.path.join(self_dir, "augmentation_strategy-swin.json")
        if "aug_strat_path" in parameters.dataset_params:
            self_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
            aug_strat_path = os.path.join(self_dir, parameters.dataset_params.aug_strat_path)

        super().__init__(parameters, aug_strat_path=aug_strat_path)

    def init_model(self):      
    # DDP broadcasts model states from rank 0 process to all other processes 
    # in the DDP constructor
    
        self.model_params.transfer_learning_params = self.transfer_learning_params
        self.model_params.save_dir = self.training_params.save_dir
        
        # init model and wrap it with DINO
        if hasattr(transformers, self.model_params.backbone_type):
            student_params = deepcopy(self.model_params)
            teacher_params = deepcopy(self.model_params)
            student_params.transformers_params.update(drop_path_rate=0.1)
            student = Classifier(student_params)
            teacher = Classifier(teacher_params)
        else:
            student, teacher = Classifier(self.model_params), Classifier(self.model_params)
            
        teacher.load_state_dict(deepcopy(student.state_dict()))    
        momentum_iters = len(self.dataloaders.trainloader) * self.training_params.epochs
        model = DINO(student, teacher, momentum_iters)
        
        if ddp_is_on():
            model = DDP(model, device_ids=[self.device_id])        
        return model
