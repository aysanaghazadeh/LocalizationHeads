import os
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

def convert_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    """
    Convert Hydra DictConfig to SimpleNamespace for backward compatibility
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Flatten nested config for backward compatibility
    flat_dict = {}
    
    # Add top-level keys
    for k, v in config_dict.items():
        if k not in ['model', 'dataset', 'experiment', 'save', 'hydra']:
            flat_dict[k] = v
    
    # Add model keys
    for k, v in config_dict.get('model', {}).items():
        flat_dict[k] = v
        if k == 'name':
            flat_dict['model_name'] = v
        elif k == 'base':
            flat_dict['model_base'] = v
        elif k == 'cache_dir':
            flat_dict['cache_dir'] = v
    
    # Add dataset keys
    for k, v in config_dict.get('dataset', {}).items():
        if k == 'name':
            flat_dict['daset'] = v
        else:
            flat_dict[k] = v
    
    # Add experiment keys
    for k, v in config_dict.get('experiment', {}).items():
        if k == 'name':
            flat_dict['exp_name'] = v
        else:
            flat_dict[k] = v
    
    # Add save options
    for k, v in config_dict.get('save', {}).items():
        flat_dict[f'save_{k}'] = v
    
    # Set device
    if 'device' in flat_dict:
        flat_dict['device'] = f"cuda:{flat_dict['device']}"
    
    # Convert boolean values from YAML
    for k, v in flat_dict.items():
        if isinstance(v, bool):
            flat_dict[k] = int(v)
    
    return SimpleNamespace(**flat_dict)

def get_config_path():
    """Get the default config path"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config") 