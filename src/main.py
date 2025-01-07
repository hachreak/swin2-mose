import argparse
import torch

from config import parse_config
from utils import load_fun, set_deterministic, print_weights
from visualize import main as vis_main
from validation import main as val_main, print_metrics as val_print_metrics, \
        load_metrics
from debug import measure_avg_time, compute_flops
from super_res.model import build_model


def parse_configs():
    parser = argparse.ArgumentParser(description='SuperRes model')
    # For training and testing
    parser.add_argument('--config',
                        default="cfgs/sen2venus_v26_8.yml",
                        help='Configuration file.')
    parser.add_argument('--phase',
                        default='train',
                        choices=['train', 'test', 'mean_std', 'vis',
                                 'plot_data', 'avg_time', 'weights',
                                 'flops'],
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
                        default='./output/sen2venus_v26_8',
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
    parser.add_argument('--eval_method',
                        default=None, type=str,
                        help='Non-DL method to use on evaluation.')
    parser.add_argument('--repeat_times',
                        type=int,
                        default=1000,
                        help='Measure times repeating model call')
    help_warm = 'Warm model calling it before starting the measure'
    parser.add_argument('--warm_times',
                        type=int,
                        default=10,
                        help=help_warm)
    parser.add_argument('--dpi',
                        type=int,
                        default=2400,
                        help="dpi in png output file.")
    parser.add_argument('--metrics_values',
                        action="store_true",
                        default=False,
                        help="Save metrics values.")

    args = parser.parse_args()
    return parse_config(args)


def main(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dataset_fun = load_fun(cfg.dataset.get(
        'load_dataset', 'datasets.sen2venus.load_dataset'))

    if cfg.phase == 'avg_time':
        print(measure_avg_time(cfg))
    elif cfg.phase == 'train':
        train_fun = load_fun(cfg.get('train', 'srgan.training.train'))
        train_dloader, val_dloader, _ = load_dataset_fun(cfg)
        train_fun = load_fun(cfg.get('train', 'srgan.training.train'))
        train_fun(train_dloader, val_dloader, cfg)
    elif cfg.phase == 'mean_std':
        if 'stats' in cfg.dataset.keys():
            cfg.dataset.pop('stats')
        _, _, concat_dloader = load_dataset_fun(cfg, concat_datasets=True)
        fun = load_fun(cfg.get(cfg.phase))
        fun(concat_dloader, cfg)
    elif cfg.phase == 'plot_data':
        _, _, concat_dloader = load_dataset_fun(cfg, concat_datasets=True)
        fun = load_fun(cfg.get(cfg.phase))
        fun(concat_dloader, cfg)
    elif cfg.phase == 'weights':
        model = build_model(cfg)
        print_weights(model, cfg)
    elif cfg.phase == 'vis':
        cfg['batch_size'] = 1
        vis_main(cfg)
    elif cfg.phase == 'flops':
        flops, macs, params, shape = compute_flops(cfg)
        print('flops', flops)
        print('macs', macs)
        print('params', params)
        print('shape', shape)
    elif cfg.phase == 'test':
        try:
            if cfg.eval_method is not None:
                raise FileNotFoundError()
            metrics = load_metrics(cfg)
        except FileNotFoundError:
            _, val_dloader, _ = load_dataset_fun(cfg, only_test=True)
            metrics = val_main(
                val_dloader, cfg, save_metrics=cfg.eval_method is None)
        val_print_metrics(metrics)


if __name__ == "__main__":
    # parse input arguments
    cfg = parse_configs()
    # fix random seed
    set_deterministic(cfg.seed)
    # run main
    main(cfg)
