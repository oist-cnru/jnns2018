import os
import copy
import importlib

import yaml
import torch


def import_str(s):
    """Import an object from a string"""
    if isinstance(s, str):
        mod_str, cls_str = s.rsplit('.', 1)
        mod = importlib.import_module(mod_str)
        return getattr(mod, cls_str)
    else:
        return s


def load_configfile(filepath):
    """Load configuration file

    Configuration files can only contain specific changes relative to another
    configuration file. Therefore, many configuration files might end up need
    to be parsed.
    """
    with open(filepath, 'r') as fd:
        params = yaml.load(fd)
    if params.get('base', None) is not None:
        base_filepath = params['base']
        if not os.path.isabs(base_filepath):
            base_filepath = os.path.join(os.path.dirname(filepath),
                                         base_filepath)
        base_params = load_configfile(base_filepath)
        # merging params
        return merge_params(base_params, params)
    return params

def merge_params(base_params, change_params):
    """Merge two set of parameters.

    Note that this is inherently an ambiguous function, so here we consider
    that dictionary keys only go two levels deep, as such:
        {key: {key2: value}, key2: value2}
    Here value2 is not a dictionary.

    :param base_params:     the base parameters
    :param change_params:  parameter that override the values of the base
                            parameters.
    """
    base_params = copy.deepcopy(base_params)
    for key, value in change_params.items():
        if key != 'base':
            if isinstance(value, dict):
                base_params.setdefault(key, {})
                for key2, value2 in value.items():
                    base_params[key][key2] = value2
            else:
                base_params[key] = value
    return base_params

def autodevice(device=None):
    """Auto-choose GPU if available and provided device is None"""
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

def yaml_pprint(data):
    return yaml.dump(data, allow_unicode=True, indent=2,
                     width=80, default_flow_style=False)

def numpify(tsr):
    return copy.deepcopy(tsr.cpu().detach().numpy())
