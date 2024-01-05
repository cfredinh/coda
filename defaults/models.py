import os

from .bases import *
from utils.transformers import *
from torch.cuda.amp import autocast


class Identity(nn.Module):
    """An identity function."""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class Classifier(BaseModel):
    """A wrapper class that provides different CNN backbones.
    
    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """
    def __init__(self, model_params, use_fc=True):
        super().__init__()
        self.use_fc = use_fc
        self.attr_from_dict(model_params)          
        
        if self.is_vit(self.backbone_type):  
            # self.transformers_params["in_chans"] = self.img_channels
            self.backbone = transformers.__dict__[self.backbone_type](**self.transformers_params, 
                                                                      pretrained=self.pretrained)
            fc_in_channels = self.backbone.num_features

        elif self.is_cnn(self.backbone_type):
            self.backbone = cnn_models.__dict__[self.backbone_type](pretrained=self.pretrained)
            # loading non-standard weights
            pretrained_type = self.cnn_params.pretrained_type if hasattr(self, "cnn_params") else "supervised"
            if self.pretrained and pretrained_type != "supervised":
                pre_cpt = download_cnn_weights(self.backbone_type, pretrained_type)
                missed_keys = self.backbone.load_state_dict(pre_cpt, strict=False)
                missing_head = set(missed_keys.missing_keys) == set(['fc.weight', 'fc.bias'])
                unexpected_keys = missed_keys.unexpected_keys == []
                is_ok = missing_head and unexpected_keys
                if not is_ok:
                    raise ValueError(f"Found unexpected keysor keys are missing: {missed_keys}")
                print_ddp(f"\033[96m Using pretrained type: {pretrained_type}\033[0m")
            fc_in_channels = self.backbone.fc.in_features
        else:
            raise NotImplementedError  
            
        self.embed_dim = fc_in_channels
        self.backbone.fc = Identity()  # removing the fc layer from the backbone (which is manually added below)

        # modify stem and last layer
        if self.use_fc:
            self.fc = nn.Linear(fc_in_channels, self.n_classes)
        self.modify_first_layer(self.img_channels, self.pretrained)            
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)   

    def forward(self, x, return_embedding=False, use_patchifier=True, return_all=False):
        with autocast(self.use_mixed_precision):
            
            if self.freeze_backbone:
                self.backbone.eval()
                
            if isinstance(x, list) and self.is_cnn(self.backbone_type):
                idx_crops = torch.cumsum(torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in x]),
                    return_counts=True,
                )[1], 0)
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx             
            else:
                if self.is_vit(self.backbone_type):
                    x_emb = self.backbone(x, use_patchifier=use_patchifier, return_all=return_all)              
                    if self.use_fc and return_all:
                        x_emb = x_emb[:, 0]
                else:
                    x_emb = self.backbone(x)                
                
            if self.use_fc:
                x = self.fc(x_emb)
            else:
                x = x_emb
            
            if return_embedding:
                return x, x_emb        
            else:
                return x
        
    def modify_first_layer(self, img_channels, pretrained):
        backbone_type = self.backbone.__class__.__name__
        if img_channels == 3:
            return

        if backbone_type == 'ResNet':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight 
         
        elif backbone_type == 'VisionTransformer':
            patch_embed_attrs = ["img_size", "patch_size", "embed_dim"]
            patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
            self.backbone.patch_embed = transformers.deit.PatchEmbed(in_chans=img_channels, **patch_defs)
            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias           
                    
        elif backbone_type == 'SwinTransformer':
            patch_embed_attrs = ["img_size", "patch_size", "embed_dim", "norm_layer"]
            patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data
            if self.backbone.patch_embed.norm is not None:
                norm_weight = self.backbone.patch_embed.norm.weight.data                
                norm_bias = self.backbone.patch_embed.norm.bias.data                
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
            self.backbone.patch_embed = transformers.swin.PatchEmbed(in_chans=img_channels, **patch_defs)
            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias      
                if self.backbone.patch_embed.norm is not None:
                    if self.backbone.patch_embed.norm.weight is not None:
                        self.backbone.patch_embed.norm.weight.data = norm_weight
                    if self.backbone.patch_embed.norm.bias is not None:
                        self.backbone.patch_embed.norm.bias.data = norm_bias
       
        else:
            raise NotImplementedError("channel modification is not implemented for {}".format(backbone_type))
