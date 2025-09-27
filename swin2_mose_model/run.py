import yaml
import torch
import pprint
import argparse

from model import Swin2MoSE


def to_shape(t1, t2):
    t1 = t1[None].repeat(t2.shape[0], 1)
    t1 = t1.view((t2.shape[:2] + (1, 1)))
    return t1


def norm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor - to_shape(mean, tensor)) / to_shape(std, tensor)


def denorm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor * to_shape(std, tensor)) + to_shape(mean, tensor)


def run_model(model, lr, stats):
    # norm fun
    lr_stats = stats['tensor_10m_b2b3b4b8']
    hr_stats = stats['tensor_05m_b2b3b4b8']

    # normalize data
    lr = norm(lr, mean=lr_stats['mean'], std=lr_stats['std'])

    sr = model(lr)

    if not torch.is_tensor(sr):
        # remove the MoE loss value
        sr, _ = sr

    # denormalize data back
    sr = denorm(sr, mean=hr_stats['mean'], std=hr_stats['std'])

    return sr


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='config-70.yml')
parser.add_argument('--weights', default='model-70.pt')

opt = parser.parse_args()

# load config
with open(opt.cfg, 'r') as f:
    cfg = yaml.safe_load(f)

# build model
model = Swin2MoSE(**cfg['super_res']['model'])

# load checkpoint
checkpoint = torch.load(opt.weights)
model.load_state_dict(checkpoint['model_state_dict'])

pprint.pprint(cfg)
print(model)

lr = torch.randn(1, 4, 128, 128)

sr = run_model(model, lr, cfg['dataset']['stats'])

print(sr.shape)
