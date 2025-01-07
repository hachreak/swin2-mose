import torch

from time import time

from utils import load_fun


def compute_flops(cfg):
    from calflops import calculate_flops

    vis = cfg.get('visualize', {})
    model = load_fun(vis.get('model'))(cfg)
    shape = tuple([cfg.batch_size, ] + vis.input_shape)

    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=shape,
            output_as_string=True,
            output_precision=4)

    return flops, macs, params, shape


def log_losses(losses, phase, writer, index):
    # Write the data during training to the training log file
    for k, v in losses.items():
        writer.add_scalar("{}/{}".format(phase, k), v.item(), index)


def _get_abs_weights_grads(model):
    return torch.cat([
            p.grad.detach().view(-1) for p in model.parameters()
            if p.requires_grad
        ]).abs()


def _get_abs_weights(model):
    return torch.cat([
        p.detach().view(-1) for p in model.parameters()
        if p.requires_grad
    ]).abs()


def _perf_measure(model, data, count, warm_count=0):
    for _ in range(warm_count):
        model(data)
    start = time()
    for _ in range(count):
        model(data)
    end = time()
    return (end - start) / count


def measure_avg_time(cfg):
    vis = cfg.get('visualize', {})
    model = load_fun(vis.get('model'))(cfg)
    x = torch.rand([cfg.batch_size, ] + vis.input_shape).to(cfg.device)
    return _perf_measure(model, x, cfg.repeat_times, cfg.warm_times)


def log_hr_stats(lr, sr, hr, writer, index, cfg):
    if cfg.get('debug', {}).get('sr_hr', False):
        def log_delta(delta, name):
            writer.add_scalar('stats/mean_{}'.format(name), delta.mean(),
                              index)
            writer.add_scalar('stats/max_{}'.format(name), delta.max(), index)

            q_0_99 = torch.quantile(delta, 0.99, interpolation='nearest')
            q_0_999 = torch.quantile(delta, 0.999, interpolation='nearest')
            writer.add_scalar('stats/q099_{}'.format(name), q_0_99, index)
            writer.add_scalar('stats/q0999_{}'.format(name), q_0_999, index)

        sf = cfg.metrics.upscale_factor
        upscale = torch.nn.Upsample(scale_factor=sf, mode='bicubic')
        log_delta((sr - upscale(lr)).abs(), 'sr_lr')
        log_delta((sr - hr).abs(), 'sr_hr')
