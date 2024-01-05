from defaults.bases import *
from torch.cuda.amp import autocast

__all__ = ['DINO']

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, barlow_loss_factor=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
        self.barlow_loss_factor = barlow_loss_factor
        if self.barlow_loss_factor:
            self.bn = nn.BatchNorm1d(out_dim, affine=False)
            self.barlow_lambda  = 1e-3 
            self.barlow_scaling_factor = out_dim + 0.25*(out_dim*(out_dim-1))*self.barlow_lambda        

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        barlow_loss = 0
        n_loss_terms = 0
        batch_size   = student_out[0].shape[0]
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
                if self.barlow_loss_factor:
                    c = self.bn(teacher_out[iq]).T @ self.bn(student_out[v])
                    # sum the cross-correlation matrix between all gpus
                    c.div_(batch_size)
                    
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = self.off_diagonal(c).pow_(2).sum()
                    barlow_loss += (on_diag + self.barlow_lambda * off_diag)                
                    
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        
        if self.barlow_loss_factor:
            barlow_loss /= n_loss_terms            
            barlow_loss /= self.barlow_scaling_factor
            total_loss = (1.0 - self.barlow_loss_factor) * total_loss + self.barlow_loss_factor * barlow_loss
        return total_loss
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = dist_average_tensor(batch_center)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    

class DINOHead(nn.Module):
    # Taken from https://github.com/facebookresearch/dino/blob/a52c63ba27ae15856a5d99e42c5a1b82daa902d8/vision_transformer.py#L314
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    
class DINO(BaseModel):
    def __init__( self, student, teacher, momentum_iters, projection_size = 4096, moving_average_decay = 0.99,
                barlow_loss_factor=0):
        super().__init__()
        self.n_classes = student.n_classes
        self.img_channels = student.img_channels
        self.transfer_learning_params = student.transfer_learning_params
        self.embed_dim = student.fc.in_features
        self.save_dir = student.save_dir
        
        dino_args = student.DINO if hasattr(student, "DINO") else {}
        projection_size = dino_args.get('projection_size', projection_size)
        moving_average_decay = dino_args.get('moving_average_decay', moving_average_decay)    
        
        self.loss_fn = DINOLoss(out_dim=projection_size, 
                                ncrops=dino_args.get('ncrops', 8), 
                                warmup_teacher_temp=dino_args.get('warmup_teacher_temp', 0.04), 
                                teacher_temp=dino_args.get('teacher_temp', 0.07),
                                warmup_teacher_temp_epochs=dino_args.get('warmup_teacher_temp_epochs', 30), 
                                nepochs=dino_args.get('nepochs', 1000), 
                                student_temp=dino_args.get('student_temp', 0.1), 
                                center_momentum=dino_args.get('center_momentum', 0.9),
                                barlow_loss_factor=dino_args.get('barlow_loss_factor', 0.)).cuda()        

        # create online and target encoders
        self.student_encoder = student
        self.teacher_encoder = teacher
        
        # create online projectors and predictors
        in_dim = student.fc.in_features
        self.student_encoder.fc = DINOHead(in_dim=in_dim, out_dim=projection_size,
                                             use_bn=False, norm_last_layer=True)
        self.teacher_encoder.fc = DINOHead(in_dim=in_dim, out_dim=projection_size,
                                             use_bn=False, norm_last_layer=True)
        self.teacher_encoder.fc.load_state_dict(deepcopy(self.student_encoder.fc.state_dict()))        
        assert modules_are_equal(student, teacher), "The Teacher and the Student must have the same initial weights"
        
        if self.transfer_learning_params.use_pretrained:
            # Getting args and paths for pretrained state
            pretrained_path = self.transfer_learning_params.pretrained_path
            pretrained_model_name = self.transfer_learning_params.pretrained_model_name
            if not pretrained_path:
                pretrained_path = os.path.join(self.save_dir, "checkpoints")
            pretrained_path = os.path.join(pretrained_path, pretrained_model_name)
            pretrained_state = state_dict_from_path(pretrained_path)
            is_dualnorm  = len([k for k in teacher.state_dict().keys() if "backbone.patch_embed.post_norm" in k]) > 0
            has_dualnorm = len([k for k in pretrained_state.keys() if "backbone.patch_embed.post_norm" in k]) > 0
            # if checkpoint does not have dual norm but model has then skip the patch weights
            if is_dualnorm and not has_dualnorm:
                pretrained_state = OrderedDict([(k,v) for k,v in pretrained_state.items() if 'patch_embed' not in k])
            
            # Clean up the state dict based on the pretrained method
            teacher_state = OrderedDict([(k.split("teacher_encoder.")[1], v) 
                                            for k,v in pretrained_state.items() 
                                            if 'teacher_encoder' in k])  
            student_state = OrderedDict([(k.split("student_encoder.")[1], v) 
                                            for k,v in pretrained_state.items() 
                                            if 'student_encoder' in k])
                
            # Load the cleaned state dict
            if teacher.backbone.dual_patch_norm:                            
                dif_keys = self.teacher_encoder.load_state_dict(teacher_state, strict=False)
                if [k for k in dif_keys.missing_keys if 'patch_embed' in k] == dif_keys.missing_keys: pass
                else: assert not dif_keys.missing_keys, "\tThe pretrained state did not much the teacher model"
                dif_keys = self.student_encoder.load_state_dict(student_state, strict=False)
                if [k for k in dif_keys.missing_keys if 'patch_embed' in k] == dif_keys.missing_keys: pass
                else: assert not dif_keys.missing_keys, "\tThe pretrained state did not much the student model"            
            else:
                self.teacher_encoder.load_state_dict(teacher_state, strict=True)
                self.student_encoder.load_state_dict(student_state, strict=True)
            print("\033[92m\tPretrained weights are loaded to DINO\033[0m")                  
        
        # freezing teacher
        self.freeze_submodel(self.teacher_encoder) 

        # init EMA
        self.ema_updater = EMA(moving_average_decay)
        self.momentum_scheduler = CosineSchedulerWithWarmup(base_value=moving_average_decay, 
                                                            final_value=1., iters=momentum_iters) 
        
        # send the BYOL wrapped model to the original model's GPU ID
        self.to(self.device_id)

    def ema_update(self, it):
        self.ema_updater.beta = self.momentum_scheduler(it)
        for online_params, target_params in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            target_params.data = self.ema_updater(online_params.detach().data, target_params.data)

    def forward(self, x, return_embedding = False, adverserial_targets = None, epoch = 0, it=0):
        
        # Forward pass               
        with autocast(self.use_mixed_precision):
            if return_embedding:
                x = x.to(self.device_id, non_blocking=True)
                return None, self.teacher_encoder.backbone(x) 

            images = [im.to(self.device_id, non_blocking=True) for im in x]
            with torch.no_grad(): 
                teacher_output = self.teacher_encoder(images[:2]).detach()  # only the 2 global views pass through the teacher
            student_output = self.student_encoder(images)
            loss = self.loss_fn(student_output, teacher_output, epoch)

            return loss.mean()
