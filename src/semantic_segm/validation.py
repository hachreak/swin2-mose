import torch

from tqdm import tqdm
from collections import OrderedDict

from utils import F1AverageMeter, load_fun
from chk_loader import load_checkpoint
from validation import get_result_filename


def validate(g_model, val_dloader, epoch, writer, mode, cfg):
    g_model.eval()
    avg_metrics = build_avg_metrics(cfg)

    with torch.no_grad():
        for j, batch in tqdm(
                enumerate(val_dloader), total=len(val_dloader),
                desc='Val Epoch: %d / %d' % (epoch + 1, cfg.epochs)):

            batch['image'] = batch['image'].to(cfg.device)
            batch['mask'] = batch['mask'].to(cfg.device)

            out = g_model(batch['image'])
            preds = out.max(-1)[1]

            for k, fun in avg_metrics.items():
                avg_metrics[k].update((preds, batch['mask']))

    if writer is not None:
        for k, v in avg_metrics.items():
            try:
                writer.add_scalar(
                    "{}/{}".format(mode, k), v.avg.item(), epoch+1)
            except RuntimeError:
                # skip if metric is a list (like f1 per class)
                pass

    g_model.train()
    return avg_metrics


def build_avg_metrics(cfg):
    return OrderedDict([
        ('f1_macro', F1AverageMeter(
            name="f1_macro", fmt=":4.4f", average='macro', cfg=cfg)),
        ('f1_micro', F1AverageMeter(
            name="f1_micro", fmt=":4.4f", average='micro', cfg=cfg)),
        ('f1_class', F1AverageMeter(
            name="f1_class", fmt=":4.4f", average=None, cfg=cfg)),
    ])


def main(val_dloader, cfg):
    model = load_eval_method(cfg)
    result = validate(
        model, val_dloader, cfg.epoch - 1, None, 'test', cfg)
    save_metrics(result, cfg)
    return result


def load_eval_method(cfg):
    vis = cfg.visualize
    model = load_fun(vis.get('model'))(cfg)
    # Load model state dict
    try:
        checkpoint = load_checkpoint(cfg)
        _, _ = load_fun(vis.get('checkpoint'))(model, checkpoint)
    except Exception as e:
        print(e)
        exit(-1)

    return model


def load_metrics(cfg):
    filename = get_result_filename(cfg)
    print('load results {}'.format(filename))
    result = torch.load(filename)
    # check if epoch corresponds
    assert result['epoch'] == cfg.epoch
    # loadl classes
    cfg.classes = result['classes']
    # build AVG objects
    avg_metrics = build_avg_metrics(cfg)
    for k, v in result['metrics'].items():
        avg_metrics[k].avg = v
    return avg_metrics


def save_metrics(metrics, cfg):
    filename = get_result_filename(cfg)
    print('save results {}'.format(filename))
    torch.save({
        'classes': cfg.classes,
        'epoch': cfg.epoch,
        'metrics': OrderedDict([
            (k, v.avg) for k, v in metrics.items()
        ])
    }, filename)
