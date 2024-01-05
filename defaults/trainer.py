import wandb
import defaults
from defaults.bases import *
import matplotlib.pyplot as plt
from .wrappers import DefaultWrapper, dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler as DS

        
class Trainer(BaseTrainer):
    """Main trainer class.

    Initializes with a DefaultWrapper instance as its input. 
    Call trainer.train() to train and validate or call trainer.test()
    Training with standard DDP: a model is trainedon multiple machines/gpus using distributed gradients. 
    """
    def __init__(self, wraped_defs):
        """Initialize the trainer instance.
        
        This function clones its attributes from the DefaultWrapper instance or generates
        them from the .json file. 
        """
        super().__init__()

        self.parameters = wraped_defs.parameters
        self.is_supervised = wraped_defs.is_supervised        
        self.training_params = self.parameters.training_params
        self.attr_from_dict(self.training_params)
        self.attr_from_dict(wraped_defs.dataloaders)
        self.epoch_steps = len(self.trainloader)
        self.total_steps = int(len(self.trainloader) * self.epochs)
        
        self.model = wraped_defs.model
        self.criterion = wraped_defs.criterion        
        self.optimizer = wraped_defs.optimizer 
        self.scheduler = wraped_defs.schedulers
        self.metric_fn = wraped_defs.metric
        
        self.org_model_state = model_to_CPU_state(self.model)
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        self.total_step = len(self.trainloader)        
        self.best_model = deepcopy(self.org_model_state)  
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.set_models_precision(self.use_mixed_precision)        
        
        
    def train(self):
        """Main training loop."""
        self.test_mode = False
        if not self.is_grid_search:
            self.load_session(self.restore_only_model)
        self.print_train_init()
        
        n_classes = self.trainloader.dataset.n_classes
        metric    = self.metric_fn(n_classes, self.trainloader.dataset.int_to_labels, mode="train")
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
            
        for self.epoch in epoch_bar:            
            self.model.train() 
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)            
            
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, metric=metric, it=it)   
                
                if self.val_every != np.inf:
                    if self.iters % int(self.val_every * self.epoch_steps) == 0: 
                        synchronize()
                        self.epoch_step()  
                        self.model.train()
                synchronize()
                
            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.model)   
                self.save_session()                         
                
        if self.is_rank0:         
            print(" ==> Training done")
        if not self.is_grid_search:
            self.evaluate()
            self.save_session(verbose=True)
        synchronize()
        
    def global_step(self, **kwargs):
        """Function for the standard forward/backward/update.
        
        If using DDP, metrics (e.g. accuracy) are calculated with dist.all_gather
        """
        self.optimizer.zero_grad()
        
        metric = kwargs['metric']        
        images, labels = kwargs['batch']
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]
        labels = labels.to(self.device_id, non_blocking=True)
        images = images.to(self.device_id, non_blocking=True) 
        
        with autocast(self.use_mixed_precision):
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()   
        
        metric.add_preds(outputs, labels) # distributed gather inside
        self.scheduler.step(self.val_target, self.val_loss)

        if not self.is_grid_search:
            if self.iters % self.log_every == 0 or self.iters == 1:
                loss = dist_average_tensor(loss)
                if self.is_rank0:
                    self.logging({'train_loss': loss.item(),
                                 'learning_rate': self.get_lr()})
                    self.logging(metric.get_value())     
                    metric.reset()                
    
    def epoch_step(self, **kwargs): 
        """Function for periodic validation, LR updates and model saving.
        
        Note that in the 2nd phase of training, the behavior is different, each model on
        each GPU is saved separately.
        """
        self.evaluate()        
        if not self.is_grid_search:
            self.save_session()                  
    
    def evaluate(self, dataloader=None, **kwargs):
        """Validation loop function.
        
        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        
        # if using knn evaluations build a feature bank of the train set
        if self.knn_eval:
            self.build_feature_bank()
            
        if not self.is_rank0: return
        self.model.eval()

        if dataloader == None:
            dataloader = self.valloader

        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.model)
            self.model.train()
            return
        
        knn_nhood     = dataloader.dataset.knn_nhood
        n_classes     = dataloader.dataset.n_classes
        target_metric = dataloader.dataset.target_metric

        if self.is_rank0:
            metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="val")
            if self.knn_eval:
                knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="knn_val")
            iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader

        val_loss = []
        feature_bank = []
        with torch.no_grad():
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    outputs, features = self.model.module(images, return_embedding=True)
                else:
                    outputs, features = self.model(images, return_embedding=True)
                    
                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())   
                        
                if self.knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank=self.feature_bank, 
                                                   feature_labels= self.targets_bank, 
                                                   knn_k=knn_nhood, knn_t=0.1, classes=n_classes, 
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels, using_knn=True)

                loss = self.criterion(outputs, labels)
                val_loss.append(loss.item())
                metric.add_preds(outputs, labels)
                
        # building Umap embeddings
        if self.log_embeddings:
            self.build_umaps(feature_bank, dataloader, labels = knn_metric.truths, mode='val')
            
        self.val_loss = np.array(val_loss).mean()
        eval_metrics = metric.get_value(use_dist=isinstance(dataloader,DS))
        if self.knn_eval:
            eval_metrics.update(knn_metric.get_value(use_dist=isinstance(dataloader,DS)))
        self.val_target = eval_metrics[f"val_{target_metric}"]

        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
                self.logging({'val_loss': round(self.val_loss, 5)})
            if self.val_target > self.best_val_target:
                self.best_val_target = self.val_target
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.model)
            if self.val_loss <= self.best_val_loss:
                self.best_val_loss = self.val_loss
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)
        self.model.train()
        
    def test(self, dataloader=None, test_all=False, **kwargs):
        """Test function.
        """
        if not self.is_rank0: return
            
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        try:
            self.load_session(self.restore_only_model)
        except:
            print("WARNING: Full checkpoint not found... Proceeding with partial model (assuming transfer learning is ON)")
        self.model.eval()
        
        if self.knn_eval:
            self.build_feature_bank()
            
        if dataloader == None:
            if test_all:
                dataloader=self.all_testloader
            else:
                dataloader=self.testloader
            
        results_dir = os.path.join(self.save_dir, 'results', self.model_name)
        metrics_path = os.path.join(results_dir, "metrics_results.json")
        check_dir(results_dir)   

        test_loss = []
        feature_bank = []
        results = edict()
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes    
        target_metric = dataloader.dataset.target_metric
        if self.is_supervised:
            metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="test")
        if self.knn_eval or not self.is_supervised:
            knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="knn_val")
        iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))
        
        with torch.no_grad():
            for images, labels in iter_bar: 
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    outputs, features = self.model.module(images, return_embedding=True)
                else:
                    outputs, features = self.model(images, return_embedding=True)
                    
                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())                      
                if self.knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank=self.feature_bank, 
                                                   feature_labels= self.targets_bank, 
                                                   knn_k=knn_nhood, knn_t=0.1, classes=n_classes,
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels, using_knn=True) 
                if self.is_supervised:
                    loss = self.criterion(outputs, labels)
                    test_loss.append(loss.item())
                    metric.add_preds(outputs, labels)
                
        if self.log_embeddings:
            self.build_umaps(feature_bank, dataloader, labels = metric.truths if self.is_supervised else knn_metric.truths, 
                        mode = 'test', wandb_logging=False)                         
        
        has_knn = False
        self.test_loss = np.array(test_loss).mean() if test_loss else None
        test_metrics = {}
        knn_test_metrics = None
        if self.is_supervised:
            test_metrics = metric.get_value(use_dist=isinstance(dataloader,DS))
        if self.knn_eval or not self.is_supervised:
            has_knn = True
            test_metrics.update(knn_metric.get_value(use_dist=isinstance(dataloader,DS)))  
        if self.is_supervised:
            self.test_target = test_metrics[f"test_{target_metric}"]
            test_metrics['test_loss'] = round(self.test_loss, 5)
        else:
            self.test_target = test_metrics.knn_test_accuracy
        
        self.model.train()
        self.set_models_precision(self.use_mixed_precision)
        save_json(test_metrics, metrics_path)
        if has_knn:
            splitted = self.model_name.split("-run_")
            if len(splitted) == 1:
                knn_mname = f"{splitted[0]}-knn"
            elif len(splitted) == 2:     
                knn_mname = f"{splitted[0]}-knn-run_{splitted[1]}"
            else:
                raise ValueError(f"Model name {self.model_name} does not follow the naming convention")
            knn_results_dir = os.path.join(self.save_dir, 'results', knn_mname)
            knn_metrics_path = os.path.join(knn_results_dir, "metrics_results.json")
            check_dir(knn_results_dir)   
            save_json(test_metrics, knn_metrics_path)
        
        print('\n',"--"*5, "{} evaluated on the test set".format(self.model_name), "--"*5,'\n')
        test_metrics = pd.DataFrame.from_dict(test_metrics, orient='index').T
        print(tabulate(test_metrics, headers = 'keys', tablefmt = 'psql'))
        print('\n',"--"*35, '\n')        

        
    def build_umaps(self, feature_bank, dataloader, labels = None, mode='', wandb_logging=True):
        if not dataloader.dataset.is_multiclass: return
        feature_bank = torch.cat(feature_bank, dim=0).numpy()
        umap_path = self.get_embedding_path(mode=mode, iters=self.iters)
        create_umap_embeddings(feature_bank, labels, 
                                   label_mapper=dataloader.dataset.int_to_labels,
                                   umap_path=umap_path)

        if wandb_logging:  
            if self.use_tensorboard:
                umap_plot = plt.imread(umap_path)
                self.logging({"umap_embeddings": [umap_plot[:,:,:3]]})
            else:
                umap_plot = Image.open(umap_path) 
                self.logging({"umap_embeddings": [wandb.Image(umap_plot, 
                                                          caption=self.model_name)]})
            
    def build_feature_bank(self, dataloader=None, **kwargs):
        """Build feature bank function.
        
        This function is stores the feature representation of the training images along with their respective labels 
        """
        
        self.model.eval()
        if dataloader is None:
            dataloader = self.fbank_loader         
        
        n_classes = dataloader.dataset.n_classes
        if self.is_rank0:
            iter_bar = tqdm(dataloader, desc='Building Feature Bank', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader
            
        with torch.no_grad():
            
            self.feature_bank = []
            self.targets_bank = []   
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    _, feature = self.model.module(images, return_embedding=True)
                else:
                    _, feature = self.model(images, return_embedding=True)
                  
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(labels)

            self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
            self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()

            synchronize()
            self.feature_bank = dist_gather(self.feature_bank, cat_dim=-1)
            self.targets_bank = dist_gather(self.targets_bank, cat_dim=-1)
        self.model.train()
        

    # FROM: Find the similarities between the batch samples and the feature bank
    def knn_predict(self, feature, feature_bank, feature_labels, 
                    knn_k: int, knn_t: float, classes: int = 10, multi_label = False):
        """Helper method to run kNN predictions on features based on a feature bank

        Args:
            feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t:

        """
    
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

        # we do a reweighting of the similarities
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        
        # convert them to probablilities
        pred_scores = pred_scores/pred_scores.sum(1).unsqueeze(1)
        
            
        return pred_scores
    