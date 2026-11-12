import torch
from torch import nn
import json
import pickle

NUM_CLASSES_DICT = {
    'face_badge_plate': 3,
    'imagenet': 1000,
    'cifar10': 10,
    'cifar100': 100,
    'food101': 101,
    'caltech101': 101,
}


def load_clip_model(model_cfg, device='cuda'):
    """Load OpenCLIP model by parsing model_cfg string."""
    import open_clip

    cfgs = model_cfg.split('_')
    arch = cfgs[0]
    pretrain = cfgs[2]

    model_map = {
        'laion400m': {
            'vitb32': ('ViT-B-32-quickgelu', 'laion400m_e32'),
            'vitb16': ('ViT-B-16', 'laion400m_e32'),
        },
        'laion2b': {
            'vitb32': ('ViT-B-32', 'laion2b_e16'),
        }
    }

    model_arch, pretrained = model_map[pretrain][arch]
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_arch, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_arch)
    model.float().to(device)
    model.eval()
    return model, preprocess, tokenizer


def _prompt_sampler(prompt_tensors, sample_by='mean'):
    """Stack prompt embeddings from prompt_tensors dict.

    prompt_tensors: dict of class_id -> {'all': tensor, 'mean': tensor}
    Returns: stacked weights tensor (num_classes, embed_dim)
    """
    num_classes = len(prompt_tensors)
    weights = torch.stack(
        [prompt_tensors[str(i)]['mean'] for i in range(num_classes)], dim=0
    )
    return weights


@torch.no_grad()
def _get_text_features(model, text_prompts, tokenizer, mode='encode'):
    """Get normalized text features for class prompts.

    text_prompts: dict of class_id -> list of prompt strings (e.g. ['a photo of a {}.'])
    Returns: prompt_tensors dict with 'all' and 'mean' keys
    """
    prompt_tensors = {}
    for class_id, prompts in text_prompts.items():
        tokenized = tokenizer(prompts).to(next(model.parameters()).device)
        features = model.encode_text(tokenized)
        features = features / features.norm(dim=-1, keepdim=True)
        mean_feat = features.mean(dim=0)
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
        prompt_tensors[str(class_id)] = {
            'all': features.cpu(),
            'mean': mean_feat.cpu()
        }
    return prompt_tensors


def set_model(args, logger):
    model, preprocess, tokenizer = load_clip_model(
        model_cfg=args.model_cfg, device=args.device
    )
    logger.info(f'Loaded model: {args.model_cfg}')
    return model, preprocess, tokenizer



def set_classifier(args, prompt_tensors, logger):

    if args.method == "dataset-cls":
        num_class = 2 # binary classification, retrieved 0 or fewshot 1
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)    
        logger.info(f'Initialized classifier head with random weights. Num of classes: {num_class}')

    elif args.cls_init == 'REAL-Prompt' or args.cls_init == 'REAL-Linear' or args.cls_init == 'text':
        weights = _prompt_sampler(prompt_tensors, sample_by='mean')
        classifier_head = MyLinear(weights=weights)
        logger.info(f'Initialized classifier head with text embedding. weights.shape: {weights.shape}')

    elif args.cls_init == 'random':
        num_class = NUM_CLASSES_DICT.get(args.dataset, 3)
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)
        logger.info(f'Initialized classifier head with random weights. Num of classes: {num_class}')
    else:
        raise NotImplementedError(f'Classifier head {args.cls_init} not implemented.')

    classifier_head.to(args.device)
    
    return classifier_head



class MyLinear(nn.Module):
    def __init__(self, weights=None, inp_dim=512, num_classes=810, bias = False):
        super(MyLinear, self).__init__()
        
        if torch.is_tensor(weights):
            self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=bias) # Set bias = False, so that we simply do Zero Shot.
            with torch.no_grad():
                self.linear.weight.copy_(weights)
            self.num_classes = weights.shape[0]
        else:
            self.linear = nn.Linear(inp_dim, num_classes, bias=bias)
            self.num_classes = num_classes
        
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.linear(x)

        # no softmax here, as we use CrossEntropyLoss which implicitly performs softmax
        # x = self.softmax(x)

        return x
    
    def update_weights(self, weights): # Cosine Similarity Validation during CLIP fine-tuning. 
        with torch.no_grad():
            self.linear.weight.copy_(weights)


def build_classifier_head(args, model, text_prompts, tokenizer):
    updated_prompt_tensors = _get_text_features(model, text_prompts, tokenizer, 'encode')
    weights = _prompt_sampler(updated_prompt_tensors, sample_by='mean')
    new_head = MyLinear(weights=weights, bias=False)
    new_head.to(args.device)
    return new_head


def save_model_ckpt(args, best_records, model, classifier_head, optimizer, scheduler, logit_scale,
                    val_acc=-1, epoch=-1, num_iter=-1):
    
    model_path = f'{args.ckpt_path}/model_bs{args.bsz}_lr-cls{args.lr_classifier}_lr-bkb{args.lr_backbone}_wd{args.wd}_epoch_{epoch}_iter_{num_iter}.pth'

    state = {}
    state['best_val_acc'] = best_records['best_val_acc']
    state['best_epoch'] = best_records['best_epoch']
    state['best_iter'] = best_records['best_iter']
    # state['best_scores'] = best_scores
    state['val_acc'] = val_acc
    state['epoch'] = epoch
    state['num_iter'] = num_iter
    if not args.freeze_visual:
        state['clip'] = model.state_dict()
    state['head'] = classifier_head.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['scheduler'] = scheduler.state_dict()
    state['logit_scale'] = logit_scale

    torch.save(state, model_path)

    return model_path

         
def save_best_model(args, best_records, best_model, best_head, best_logit_scale,
                    test_acc, best_tau, best_tau_test_acc, wsft_test_acc,
                    best_tau_head, wsft_backbone, wsft_head, stage=1):
    
    best_epoch = best_records['best_epoch']
    best_iter = best_records['best_iter']
    model_path = f'{args.output_dir}/stage{stage}_model_best-epoch_{best_epoch}_best.pth'
    
    # save scores of the best model to a json file
    save_path = f'{args.output_dir}/stage{stage}_val_scores_best.json'
    with open(save_path, 'w') as f:
        json.dump(best_records['best_scores'], f, indent=4)
    
    save_path = f'{args.output_dir}/stage{stage}_val_confusion_matrix_best.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(best_records['best_confusion_matrix'], f)

    state = {}
    state['best_val_acc'] = best_records['best_val_acc']
    state['best_epoch'] = best_records['best_epoch']
    state['best_iter'] = best_records['best_iter']
    # state['best_scores'] = best_scores
    # if not args.freeze_visual:
    state['clip'] = best_model.state_dict()
    state['head'] = best_head.state_dict()
    state['logit_scale'] = best_logit_scale
    state['test_acc'] = round(test_acc, 3)
    state['best_tau'] = best_tau
    state['best_tau_test_acc'] = round(best_tau_test_acc, 3)
    state['wsft_test_acc'] = round(wsft_test_acc,3)
    state['best_tau_head'] = best_tau_head.state_dict() if best_tau_head is not None else None
    state['wsft_head'] = wsft_head.state_dict() if wsft_head is not None else None
    state['wsft_backbone'] = wsft_backbone.state_dict() if wsft_backbone is not None else None

    torch.save(state, model_path)

    return model_path


def save_test_scores(scores, confusion_matrix, output_dir, tag, stage=1):

    # save scores to a json file
    save_path = f'{output_dir}/stage{stage}_{tag}_scores.json'
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=4)

    # save the confusion matrix
    save_path = f'{output_dir}/stage{stage}_{tag}_confusion_matrix.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(confusion_matrix, f)    

def save_head_weights(classifier_head, output_dir, tag):
    save_path = f'{output_dir}/{tag}_head_weights.pth'
    torch.save(classifier_head.state_dict(), save_path)


# ==============================================================================
# SimSwap Generator (Face Privacy Protection)
# ==============================================================================


class _InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class _ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super().__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x


class _ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type='reflect'):
        super().__init__()
        activation = nn.ReLU(True)
        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), _InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = _ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), _InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = _ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents):
        y = self.conv1(x)
        y = self.style1(y, dlatents)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents)
        return x + y


class SimSwapGenerator(nn.Module):
    """AdaIN-based face swap generator for privacy-preserving face replacement.

    Architecture: Encoder → AdaIN ResBlocks ← identity latent → Decoder.
    Input: aligned face image [B, 3, crop, crop].
    Output: swapped face image [B, 3, crop, crop] in [0, 1].
    """

    def __init__(self, latent_size=512, n_blocks=9, crop_size=224):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer = nn.BatchNorm2d

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            norm_layer(64), activation)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128), activation)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256), activation)
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512), activation)

        BN = []
        for _ in range(n_blocks):
            BN.append(_ResnetBlock_Adain(512, latent_size=latent_size,
                                          padding_type='reflect'))
        self.BottleNeck = nn.Sequential(*BN)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation)
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh())

    def forward(self, x, dlatents):
        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        x = self.down3(skip3)
        for layer in self.BottleNeck:
            x = layer(x, dlatents)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        return (x + 1) / 2


def load_simswap_generator(ckpt_path, crop_size=224, device='cuda'):
    """Load SimSwap generator from checkpoint.

    Args:
        ckpt_path: path to the generator checkpoint (e.g. 90000_net_G.pth).
        crop_size: face alignment/swap resolution (224 or 512).
        device: 'cuda' or 'cpu'.

    Returns:
        SimSwapGenerator in eval mode.
    """
    netG = SimSwapGenerator(latent_size=512, n_blocks=9, crop_size=crop_size)
    netG = netG.float().to(device)
    netG.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ['netG', 'G', 'model', 'generator']:
            if key in ckpt:
                ckpt = ckpt[key]
                break
    netG.load_state_dict(ckpt, strict=False)
    return netG

