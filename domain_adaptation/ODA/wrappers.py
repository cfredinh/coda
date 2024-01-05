from defaults import *
from utils.helpfuns import *
from self_supervised.BYOL import *
from self_supervised.DINO import *
from .models import Ymodel

__all__ = ['TestTimeWrapper', 'TestTimeSSLWrapper']

class TestTimeWrapper(DefaultWrapper):
    def __init__(self, parameters, aug_strat_path=None):
        super().__init__(parameters, aug_strat_path=aug_strat_path)
        self.is_supervised = True

    def init_model(self):      
        
        # Init some defs
        self.model_params.ODA.foundation_params.n_classes    = self.model_params.n_classes
        self.model_params.ODA.foundation_params.img_channels = self.model_params.img_channels
        self.model_params.ODA.main_model_params = {"transfer_learning_params": self.transfer_learning_params}
        foundation_params = self.model_params.ODA.foundation_params
        self.model_params.save_dir = self.training_params.save_dir
                
        # INIT SSL and foundation model
        print_ddp(f"\033[94m\tInit foundation model\033[0m")
        foundation_model = Classifier(foundation_params, use_fc=False)
        
        # INIT main model
        print_ddp("\033[94m\tInit main model\033[0m")
        main_model = Classifier(self.model_params)       
        
        # INIT dual model
        model = Ymodel(self.model_params, foundation_model, main_model)
                
        if ddp_is_on():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
class TestTimeSSLWrapper(DINOWrapper):

    def __init__(self, parameters, aug_strat_path=None, ssl_type=None):
        self.ssl_type = ssl_type
        if self.ssl_type in ['dino', 'DINO']:
            aug_strat_path = os.path.dirname(os.path.abspath(inspect.getfile(DINOWrapper)))
        aug_strat_path = os.path.join(aug_strat_path, "augmentation_strategy.json")

        # Save the last checkpoint (we don't know which the best one is)
        parameters.training_params.save_best_model = False
        parameters.model_params.save_dir = parameters.training_params.save_dir
        
        # Init DINO
        print_ddp("USING AUGMENTATION STRATEGY FOUND AT: ", aug_strat_path)
        super().__init__(parameters, aug_strat_path=aug_strat_path)
        # set the validation interval
        self.training_params.val_every = np.inf
        # Always start with a SSL pre-trained model
        self.transfer_learning_params.use_pretrained = True
                
            
    def init_model(self):      
        self.model_params.transfer_learning_params = self.transfer_learning_params
        self.model_params.save_dir = self.training_params.save_dir        
        if self.ssl_type in ['dino', 'DINO']:
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
        else:
            raise NotImplementedError(f"{self.ssl_type} is not implemented")

        return model            
        
    def init_dataloaders(self, collate_fn=None):
        
        DataSet = self.dataset_mapper.get(self.dataset_params.dataset, False)
        assert DataSet, "Dataset not found - Plese select one of the following: {}".format(list(self.dataset_mapper.keys()))
        
        if self.test_all:
            trainset = DataSet(self.dataset_params, mode='train')
            valset   = DataSet(self.dataset_params, mode='eval')
            testset  = DataSet(self.dataset_params, mode='test')
            all_test = DataSet(self.dataset_params, mode='test')
            all_test.data = trainset.data + valset.data + testset.data
            
            trainset.data      = all_test.data

        else:
            # define dataset params and dataloaders  
            trainset = DataSet(self.dataset_params, mode='train')
            testset  = DataSet(self.dataset_params, mode='test')
            # get the train augmentations but the test data for ODA 
            trainset.data      = testset.data
            trainset.dataframe = testset.dataframe
        
        trainset.num_augmentations = 2     
        
        #register task defs
        self.task          = trainset.task
        self.is_multiclass = trainset.is_multiclass           

        # define distributed samplers etc
        if ddp_is_on():
            do_shuffling = self.dataloader_params['trainloader']['shuffle']
            train_sampler = DS(trainset, num_replicas=self.visible_world, rank=self.device_id)
            self.dataloader_params['trainloader']['shuffle'] = False
        else:
            train_sampler = None
            do_shuffling = self.dataloader_params['trainloader']['shuffle']
        
        trainLoader = DataLoader(trainset, **self.dataloader_params['trainloader'], sampler=train_sampler) 
        self.dataloader_params['trainloader']['shuffle'] = do_shuffling # Making sure that shuffling is ON again        
        
        return edict({'trainloader': trainLoader, 
                       'valloader' : [],
                       'testloader' : [],
                       'fbank_loader' : []})
