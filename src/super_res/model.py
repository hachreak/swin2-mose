
def build_model(cfg):
    version = cfg.super_res.get('version', 'v1')
    print('load super_res {}'.format(version))

    if version == 'v1':
        from .network_swinir import SwinIR as SRModel
    elif version == 'v2':
        from .network_swin2sr import Swin2SR as SRModel
    elif version == 'swinfir':
        from .swinfir_arch import SwinFIR as SRModel
    elif version == 'dat':
        from .dat_arch import DAT as SRModel

    model = SRModel(**cfg.super_res.model).to(cfg.device)

    return model
