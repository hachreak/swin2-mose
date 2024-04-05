from torch import optim


def build_optimizer(model, cfg):
    o_cfg = cfg

    optimizer = optim.Adam(model.parameters(),
                           o_cfg.optim.learning_rate,
                           o_cfg.optim.model_betas,
                           o_cfg.optim.model_eps,
                           o_cfg.optim.model_weight_decay)

    return optimizer
