import torch
import torch.nn as nn
import copy

from models.vit_inflora_trans import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.zoo import CodaPrompt

class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)

    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)
        
        return x, prompt_loss
    
    def interface_old(self, x, task_id, bases=None, types=None, register_blk=-1):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if bases is not None and len(bases) > 0 and len(types) > 0:
                x = blk.interface_old(x, task_id, bases[i], types[i], register_blk==i)
            else:
                x = blk(x, task_id, register_blk==i)

        x = self.norm(x)
        
        return x



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    pretrained_cfg['file'] = "./pretrained/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
    print(pretrained_cfg['file'])
    # exit()
    model = build_model_with_cfg(
        ViT_lora_co, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        # pretrained_custom_load='npz' in pretrained_cfg['url'],
        pretrained_custom_load='npz' in pretrained_cfg['file'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        self.class_num = 1
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None, bases=None, types=None):
        if task == None:
            if bases == None:
                image_features, _ = self.image_encoder(image, self.numtask-1)
            else:
                image_features = self.image_encoder.interface_old(image, self.numtask, bases, types)
        else:
            if bases == None:
                image_features, _ = self.image_encoder(image, task)
            else:
                image_features = self.image_encoder.interface_old(image, task+1, bases, types)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def extract_vector_old(self, image, task=None, bases=None, types=None):
        if task == None:
            image_features = self.image_encoder.interface_old(image, self.numtask-1, bases, types)
        else:
            image_features = self.image_encoder.interface_old(image, task, bases, types)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)
        # for prompts in [self.classifier_pool[self.numtask-1]]:
        #     logits.append(prompts(image_features))
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits
    
    def interface_old(self, image, task_id = None):
        with torch.no_grad():
            image_features = self.image_encoder.interface_old(image, task_id=self.numtask-1 if task_id is None else task_id)

            image_features = image_features[:,0,:]
            image_features = image_features.view(image_features.size(0),-1)

            logits = []
            for prompt in self.classifier_pool[:self.numtask-1]:
                logits.append(prompt(image_features))

            logits = torch.cat(logits,1)
        return logits
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:,0,:]
            image_features = image_features.view(image_features.size(0),-1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits,0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits
    
    def interface3(self, image, task_ids):
        image_features = []
        for index in range(len(task_ids)):
            image_feature, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item(), trans_knowledge=True, train=True)
            image_feature = image_feature[:,0,:]
            image_feature = image_feature.view(image_feature.size(0),-1)

            image_features.append(image_feature)

        image_features = torch.cat(image_features,0)
        
        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
