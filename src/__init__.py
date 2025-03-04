from omegaconf import OmegaConf

def get_norm_name(model_cfg):
    norm_name = ''
    if 'wrn' in model_cfg.name:
        if model_cfg.backbone.norm_type == 'bn':
            norm_name = '_bn'
    return norm_name

OmegaConf.register_new_resolver("norm_name", lambda x: get_norm_name(x))