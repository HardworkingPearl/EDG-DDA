# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import sys

sys.path.append('..')
sys.path.append('../..')

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_start = time.time()
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="DANN")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index')
    parser.add_argument('--steps', type=int, default=5001,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    # parser.add_argument('--holdout_fraction', type=float, default=0.001)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()
    print("args:", args)

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(
        os.path.join(args.output_dir, args.dataset + '_' + args.algorithm + '_' + str(args.test_envs) + '_out.txt'))
    sys.stderr = misc.Tee(
        os.path.join(args.output_dir, args.dataset + '_' + args.algorithm + '_' + str(args.test_envs) + '_err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    if not os.path.exists(args.output_dir + "/results"):
        os.makedirs(args.output_dir + "/results")
    random_string = str(random.randint(10, 99))
    t = time.localtime()
    time_str = str(t.tm_year) + "0" + str(t.tm_mon) + str(t.tm_mday) + str(t.tm_hour)
    if args.hparams:
        file_name = args.output_dir + "/results/" + args.algorithm + "_" + args.dataset + "_" + str(args.test_envs) + "_" + args.hparams + "_" + time_str + random_string + ".txt"
    else:
        file_name = args.output_dir + "/results/" + args.algorithm + "_" + args.dataset + "_" + str(
            args.test_envs) + "_" + time_str + random_string + ".txt"

    file_name = file_name.replace("\"", "")
    with open(file_name, 'a') as f:
        f.write(str(args))
        f.write(str(hparams))
        f.close()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        print("env_i:", env_i, len(env))

        out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:  # this is for da, some unlabelled data for adaptations
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))
        print("out:", len(out), " in_:", len(in_), "uda:", len(uda))  

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda.env = env_i
            uda_splits.append((uda, uda_weights))
    print("in_splits:", len(in_splits))  
    print("out_weights:", len(out_splits))  
    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    isDebug = True if sys.gettrace() else False
    n_worker = 0  # if isDebug else dataset.N_WORKERS
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        domain=i,
        num_workers=n_worker)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=n_worker)
        for i, (env, env_weights) in enumerate(uda_splits)
        if env.env in args.test_envs]  
    if args.dataset == "EDGOcularDisease" or args.dataset=="EDGCaltran":
        eval_splits = (in_splits[-3], out_splits[-3], in_splits[-2], out_splits[-2],in_splits[-1], out_splits[-1])
        eval_loaders = [FastDataLoader(  
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=n_worker)
            for env, _ in eval_splits]
    else:
        eval_splits = (in_splits + out_splits + [uda_split for uda_split in uda_splits if uda_split[0].env in args.test_envs])
        eval_loaders = [FastDataLoader(  
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=n_worker)
            for env, _ in eval_splits]

    eval_weights = [None for _, weights in eval_splits]
    print("eval_weights:", len(eval_weights))
    print("eval_weights:", eval_weights[0])
    if args.dataset == "EDGOcularDisease":
        eval_loader_names = [f'env{7}_in', f'env{7}_out', f'env{8}_in', f'env{8}_out', f'env{9}_in', f'env{9}_out']
        print("eval_loader_names:", eval_loader_names)
    elif args.dataset == "EDGCaltran":
        eval_loader_names = [f'env{43}_in', f'env{43}_out', f'env{44}_in', f'env{44}_out', f'env{45}_in', f'env{45}_out']
        print("eval_loader_names:", eval_loader_names)
    else:
        eval_loader_names = ['env{}_in'.format(i)
                             for i in range(len(in_splits))]
        print("eval_loader_names:", eval_loader_names)
        eval_loader_names += ['env{}_out'.format(i)
                              for i in range(len(out_splits))]
        print("eval_loader_names:", eval_loader_names)
        eval_loader_names += ['env{}_uda'.format(uda_split[0].env)
                              for uda_split in uda_splits if uda_split[0].env in args.test_envs]
        print("eval_loader_names:", eval_loader_names)



    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    print("algorithm_class:", algorithm_class)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    # n_steps = args.steps or dataset.N_STEPS
    # n_steps = max(args.steps, dataset.N_STEPS)
    n_steps = args.steps
    print("total steps:", n_steps, "args.steps:", args.steps)
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ


    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))



    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        if hparams['dm_idx']:
            minibatches_device = [(x.to(device), y.to(device), d.to(device))
                                  for x, y, d in next(train_minibatches_iterator)]
        else:
            minibatches_device = [(x.to(device), y.to(device))
                                  for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            if hparams['dm_idx']:
                uda_device = [[x.to(device), d.to(device)]
                              for x, _, d in next(uda_minibatches_iterator)]
            else:
                uda_device = [x.to(device)
                              for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                
                acc = misc.accuracy(algorithm, loader, weights, device, args.algorithm)
                results[name + '_acc'] = acc
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            with open(file_name, 'a') as f:
                f.write("\n")
                for k in sorted(results):
                    if "acc" in k or "loss" in k or "step" in k:
                        f.write(k + ": " + str(results[k]) + " ")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint('model_step{}.pkl'.format(step))

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    print("running time: ", time.time() - train_start)
