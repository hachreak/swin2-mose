import yaml
import os

from easydict import EasyDict
from functools import reduce


class DotDict(EasyDict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v

    def __setitem__(self, k, v):
        if isinstance(k, str) and '.' in k:
            k = k.split('.')
        if isinstance(k, (list, tuple)):
            last = reduce(lambda d, kk: d[kk], k[:-1], self)
            last[k[-1]] = v
            return
        return super().__setitem__(k, v)

    def __getitem__(self, k):
        if isinstance(k, str) and '.' in k:
            k = k.split('.')
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    #  def get(self, k, default=None):
    #      if isinstance(k, str) and '.' in k:
    #          try:
    #              return self[k]
    #          except KeyError:
    #              return default
    #      return super().get(k, default)

    def update(self, mydict):
        for k, v in mydict.items():
            self[k] = v


def parse_config(args):
    print('load config file ', args.config)
    cfg = EasyDict(load_config(args.config))

    var_args = vars(args)

    # check batch_size to overwrite only if defined
    if var_args['batch_size'] is None:
        del var_args['batch_size']

    cfg.update(var_args)

    # backup the config file
    os.makedirs(cfg.output, exist_ok=True)
    with open(os.path.join(cfg.output, 'cfg.yml'), 'w') as bkfile:
        yaml.dump(cfg, bkfile, default_flow_style=False)

    # load from node if exists
    load_node_dataset(cfg)
    # load from nvme if exists
    load_nvme_dataset(cfg)

    return cfg


def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('__base__')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        dirname = os.path.split(path)[0]
        cfg = load_config(os.path.join(dirname, inherit_from), default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def load_nvme_dataset(cfg):
    root_path_nvme = cfg.dataset.root_path.rstrip('/') + '_nvme'
    if os.path.exists(root_path_nvme):
        # update path to read from nvme disk
        cfg.dataset.root_path = root_path_nvme
        print('load dataset from {}'.format(cfg.dataset.root_path))


def load_node_dataset(cfg):
    root_path_node = cfg.dataset.root_path.rstrip('/') + '_node'
    if os.path.exists(root_path_node):
        # update path to read from nvme disk
        cfg.dataset.root_path = root_path_node
        print('load dataset from {}'.format(cfg.dataset.root_path))
