import argparse
import torch

from config import parse_config
from datasets.seasonet import load_dataset
from utils import load_fun, set_deterministic
from validation import print_metrics as val_print_metrics
from semantic_segm.validation import load_metrics, main as val_main
from semantic_segm.visualize import main as vis_main


def parse_configs():
    parser = argparse.ArgumentParser(description='Semantic Segmentation model')
    # For training and testing
    parser.add_argument('--config',
                        default="cfgs/seasonet_v1.yml",
                        help='Configuration file.')
    parser.add_argument('--phase',
                        default='test',
                        choices=['train', 'test', 'mean_std', 'vis'],
                        help='Training or testing or play phase.')
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        metavar='B',
                        help='Batch size. If defined, overwrite cfg file.')
    help_num_workers = 'The number of workers to load dataset. Default: 0'
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        metavar='N',
                        help=help_num_workers)
    parser.add_argument('--output',
                        default='./output/demo',
                        help='Directory where save the output.')
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        metavar='N',
                        help='number of epoches (default: 50)')
    help_snapshot = 'The epoch interval of model snapshot (default: 10)'
    parser.add_argument('--snapshot_interval',
                        type=int,
                        default=1,
                        metavar='N',
                        help=help_snapshot)
    parser.add_argument("--num_images",
                        type=int,
                        default=10,
                        help="Number of images to plot")
    parser.add_argument('--hide_sr',
                        action='store_true',
                        default=False)
    parser.add_argument('--dpi',
                        type=int,
                        default=2400,
                        help="dpi in png output file.")

    args = parser.parse_args()
    return parse_config(args)


if __name__ == "__main__":
    # parse input arguments
    cfg = parse_configs()
    # set random seed
    set_deterministic(cfg.seed)

    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.phase == 'train':
        train_dloader, val_dloader, _ = load_dataset(cfg)
        train_fun = load_fun(cfg.train)
        train_fun(train_dloader, val_dloader, cfg)
    elif cfg.phase == 'mean_std':
        if 'stats' in cfg.dataset.keys():
            cfg.dataset.pop('stats')
        _, _, concat_dloader = load_dataset(cfg, concat_datasets=True)
        fun = load_fun(cfg.get(cfg.phase))
        fun(concat_dloader, cfg)
    elif cfg.phase == 'vis':
        vis_main(cfg)
    elif cfg.phase == 'test':
        try:
            metrics = load_metrics(cfg)
        except FileNotFoundError:
            # validate if not already done
            _, val_dloader, _ = load_dataset(cfg)
            metrics = val_main(val_dloader, cfg)
        # print metrics
        val_print_metrics(metrics)
