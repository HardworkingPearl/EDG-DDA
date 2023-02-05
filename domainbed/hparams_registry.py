# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST', 'DenseDomainRotatedMNIST', 'EDGPortrait', 'EDGRotatedMNIST', "EDGSine", "EDGForestCover", "EDGEvolCircle"] # , "EDGOcularDisease"]

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', True, lambda r: True)  
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    # _hparam('class_balanced', False, lambda r: False)
    _hparam('class_balanced', False, lambda r: False)	
    if dataset != 'EDGSine':
        _hparam('nonlinear_classifier', True,
                lambda r: bool(r.choice([False, False])))  

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))													  


    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
    elif algorithm == "MetaAug":
        _hparam('mldg_beta', 1., lambda r: 10 ** r.uniform(-1, 1))
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD":
        _hparam('mmd_gamma', 0.001, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "CORAL":
        _hparam('mmd_gamma', 1, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "DDA":
        _hparam('dm_idx', True, lambda r: True)
        _hparam("tau_temp", 2, lambda r: r.choice([0.1, 0.01, 0.001]))
        _hparam("clip", 0.05, lambda r: r.choice([1, 0.5, 0.1, 0.01]))

        if dataset == "EDGRotatedMNIST":
            _hparam('train_step', 1, lambda r: r.choice([1, 3, 5]))
            _hparam('test_step', 5, lambda r: r.choice([1, 3, 5]))
            _hparam("alpha", 0.01, lambda r: r.choice([0.1, 0.01, 0.001]))  
            _hparam("lambda", 0.5, lambda r: r.choice([0.1, 0.01, 0.001]))
            _hparam('beta', 0.005, lambda r: r.choice([0.01, 0.05, 0.001]))
            _hparam('attn_width', 128, lambda r: 32)
            _hparam('attn_depth', 3, lambda r: 3)
        elif dataset == "EDGEvolCircle":
            _hparam('train_step', 1, lambda r: r.choice([1, 3, 5]))
            _hparam('test_step', 5, lambda r: r.choice([1, 3, 5]))
            _hparam("alpha", 2, lambda r: r.choice([0.1, 0.01, 0.001])) 
            _hparam("lambda", 0.5, lambda r: r.choice([0.1, 0.01, 0.001]))
            _hparam('beta', 0.001, lambda r: r.choice([0.01, 0.05, 0.001]))
            _hparam('attn_width', 4, lambda r: 32)
            _hparam('attn_depth', 3, lambda r: 3)
        elif dataset == "EDGPortrait":
            _hparam('train_step', 5, lambda r: r.choice([1, 3, 5]))
            _hparam('test_step', 10, lambda r: r.choice([1, 3, 5]))
            _hparam("alpha", 0.005, lambda r: r.choice([0.1, 0.01, 0.001]))  # inner
            _hparam("lambda", 0.8, lambda r: r.choice([0.1, 0.01, 0.001]))
            _hparam('beta', 0.005, lambda r: r.choice([0.01, 0.05, 0.001]))  # outer
            _hparam('attn_width', 128, lambda r: 32)
            _hparam('attn_depth', 3, lambda r: 3)	
    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.
    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(int(r.choice([32, 64]))))  
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    elif dataset == "EDGOcularDisease":
        _hparam('batch_size', 8, lambda r: int(2 ** r.uniform(3, 5)))
    elif dataset == "EDGEvolCircle":
        _hparam('batch_size', 64, lambda r: 128)
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))

    if dataset == "EDGRotatedMNIST":
        _hparam('mlp_width', 128, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))  # 5
        _hparam('mlp_dropout', 0, lambda r: 0)
        _hparam("env_sample_number", 800, lambda r: int(r.choice([300, 400, 500, 800])))
        _hparam("env_distance", 15, lambda r: int(r.choice([10, 15, 30])))
        _hparam('total_sample_number', 0, lambda r: 0)

    if dataset == "EDGPortrait":
        _hparam('mlp_dropout', 0, lambda r: 0)
        _hparam("env_sample_number", 500, lambda r: int(r.choice([300, 400, 500, 800])))
        _hparam("env_distance", 10, lambda r: int(r.choice([10, 15, 30])))
        _hparam("env_number", 11, lambda r: int(r.choice([10, 15, 30])))
        _hparam("env_sample_ratio", 0.2, lambda r: r.choice([0.2, 0.4, 0.6, 0.8, 1]))
        _hparam('mlp_width', 128, lambda r: 32)
        _hparam('mlp_depth', 3, lambda r: 3)  
        _hparam('total_sample_number', 0, lambda r: 0)

    if dataset == "EDGForestCover":
        _hparam("env_number", 10, lambda r: 10)
        _hparam("env_distance", 1, lambda r: 1)
        _hparam('mlp_dropout', 0, lambda r: 0)
        _hparam('mlp_depth', 1, lambda r: int(r.choice([1])))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(1, 1)))

    if dataset == "EDGOcularDisease" or dataset == "EDGCaltran":
        _hparam("env_number", 10, lambda r: 10)
        _hparam("mlp_width", 512, lambda r: 10)
        _hparam('mlp_depth', 1, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0, lambda r: 0)

    if dataset == "EDGEvolCircle":
        _hparam("env_number", 30, lambda r: 10)
        _hparam("env_distance", 10, lambda r: 10)
        _hparam("env_sample_number", 1000, lambda r: 1000)
        _hparam("env_sample_ratio", 0.5, lambda r: 0.5)
        _hparam("pure_liner", True, lambda r: True)
        _hparam('mlp_width', 4, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))  
        _hparam('mlp_dropout', 0, lambda r: 0)


    if dataset in ['EDGCircle', 'EDGSine']:

        _hparam('pure_liner', False, lambda r: bool(r.choice([True, True])))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([1])))
        _hparam('mlp_dropout', 0, lambda r: 0)
        if dataset == "EDGCircle":
            _hparam("env_number", 30, lambda r: 30)
            _hparam("env_distance", 6, lambda r: 6)
        if dataset == "EDGSine":
            _hparam('mlp_width', 16, lambda r: int(2 ** r.uniform(1, 1)))
            _hparam("env_number", 11, lambda r: 10) 
            _hparam("env_distance", 10, lambda r: 10)
            _hparam("env_sample_number", 50, lambda r: 1000)
            _hparam("env_sample_ratio", 0.5, lambda r: 0.5)
            _hparam("nonlinear_classifier", True, lambda r: False)

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))


    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams


def default_hparams(algorithm, dataset):  # this function to choose the default one
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):  # this function to choose the latter random one
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
