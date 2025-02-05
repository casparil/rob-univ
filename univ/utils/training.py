import argparse
import dill
import time
import torch
import numpy as np
import os
from loguru import logger

from cox import utils
from tqdm import tqdm
from robustness import train
from univ.rob import datasets, defaults


def fill_args(args_dict: dict):
    """
    Takes a dictionary containing arguments to be used for training and fills the remaining necessary arguments to train
    a model with these parameters using the robustness library.

    :param args_dict: The initial dictionary containing the required pre-set arguments.
    :return: The argument parameters where all other required values are filled with defaults based on the dataset.
    """
    args = utils.Parameters(args_dict)
    assert args.dataset is not None
    ds = datasets.DATASETS[args.dataset]
    args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, ds)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, ds)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds)
    return args


def get_kwargs(args: argparse.Namespace, dataset: str = 'imagenet', arch: str = 'vit'):
    """
    Extracts the arguments required for training vision models from the given namespace and fills missing parameters
    required by the robustness library.

    :param args: The object containing the parameters required for training.
    :param dataset: The dataset used for training, default: 'imagenet'.
    :param arch: The model architecture to train, default: 'vit'.
    :return: A dictionary of parameters required by the robustness library for training.
    """
    if bool(args.train):
        adv_eval = args.adv
    else:
        adv_eval = 1

    kwargs = {
        'out_dir': args.save,
        'adv_train': args.adv,
        'constraint': args.constraint,
        'eps': args.eps,
        "lr": args.model_lr,
        'attack_lr': args.lr,
        'attack_steps': args.steps,
        'dataset': dataset,
        'data': args.data,
        'arch': arch,
        'workers': args.workers,
        'epochs': args.epochs,
        'adv_eval': adv_eval,
        'step_lr': args.steplr,
        'batch_size': args.batch,
        'save_ckpt_iters': args.ckpt
    }

    return fill_args(kwargs)


# Code for the following methods taken from https://github.com/MadryLab/robustness
def eval_model(args, model, loader, store, devices=None):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    train.check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None:
        store.add_table(train.consts.LOGS_TABLE, train.consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    assert not hasattr(model, "module"), "model is already in DataParallel."
    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices, output_device=devices)
    else:
        model = torch.nn.DataParallel(model)

    prec1, nat_loss = _model_loop(args, 'val', loader,
                                  model, None, 0, False, writer, devices[0])
    torch.cuda.empty_cache()

    adv_prec1, adv_loss = float('nan'), float('nan')
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if train.has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if train.has_attr(args, 'attack_lr') else None
        adv_prec1, adv_loss = _model_loop(args, 'val', loader,
                                          model, None, 0, True, writer, devices[0])
    torch.cuda.empty_cache()
    log_info = {
        'epoch': 0,
        'nat_prec1': prec1.cpu().item(),
        'adv_prec1': adv_prec1.cpu().item(),
        'nat_loss': nat_loss,
        'adv_loss': adv_loss,
        'train_prec1': float('nan'),
        'train_loss': float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[train.consts.LOGS_TABLE].append_row(log_info)
    return log_info


def train_model(args, model, loaders, *, checkpoint=None, dp_device_ids=None,
                store=None, update_params=None, disable_no_grad=False, devices=None, current_epoch=0):
    """
    Main function for training a model.

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_lr_multplier (str)
                If given, use a custom LR schedule, formed by multiplying the
                    original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
            lr_interpolation (str)
                How to drop the learning rate, either ``step`` or ``linear``,
                    ignored unless ``custom_lr_multiplier`` is provided.
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            custom_eps_multiplier (str, *required if adv_train or adv_eval*)
                If given, then set epsilon according to a schedule by
                multiplying the given eps value by a factor at each epoch. Given
                in the same format as ``custom_lr_multiplier``, ``[(epoch,
                MULTIPLIER)..]``
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            custom_accuracy (function)
                If given, should be a function that takes in model outputs
                and model targets and outputs a top1 and top5 accuracy, will
                displayed instead of conventional accuracies
            regularizer (function, optional)
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)`
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        dp_device_ids (list|None) : if not ``None``, a list of device ids to
            use for DataParallel.
        store (cox.Store) : a cox store for logging training progress
        update_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
        disable_no_grad (bool) : if True, then even model evaluation will be
            run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
    """
    # Logging setup
    torch.cuda.empty_cache()
    writer = store.tensorboard if store else None
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
    if store is not None:
        store.add_table(train.consts.LOGS_TABLE, train.consts.LOGS_SCHEMA)

    # Reformat and read arguments
    train.check_required_args(args)  # Argument sanity check
    for p in ['eps', 'attack_lr', 'custom_eps_multiplier']:
        setattr(args, p, eval(str(getattr(args, p))) if train.has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None:
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = train.make_optimizer_and_schedule(args, model, checkpoint, update_params)

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = torch.nn.DataParallel(model, device_ids=devices, output_device=devices)

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[prec1_key] if prec1_key in checkpoint \
            else _model_loop(args, 'val', val_loader, model, None, start_epoch - 1, args.adv_train, writer=None)[0]
    else:
        start_epoch = current_epoch
    torch.cuda.empty_cache()

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_prec1, train_loss = _model_loop(args, 'train', train_loader,
                                              model, opt, epoch, args.adv_train, writer, devices[0])
        last_epoch = (epoch == (args.epochs - 1))
        torch.cuda.empty_cache()

        # evaluate on validation set
        sd_info = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'schedule': (schedule and schedule.state_dict()),
            'epoch': epoch + 1,
            'amp': train.amp.state_dict() if args.mixed_precision else None,
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                              store.path, filename)
            torch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            ctx = torch.enable_grad() if disable_no_grad else torch.no_grad()
            with ctx:
                prec1, nat_loss = _model_loop(args, 'val', val_loader, model,
                                              None, epoch, False, writer, devices[0])
                torch.cuda.empty_cache()

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                                                      model, None, epoch, True, writer, devices[0])
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)
            torch.cuda.empty_cache()

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            # log every checkpoint
            log_info = {
                'epoch': epoch + 1,
                'nat_prec1': prec1,
                'adv_prec1': adv_prec1,
                'nat_loss': nat_loss,
                'adv_loss': adv_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[train.consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(train.ckpt_at_epoch(epoch))

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(train.consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(train.consts.CKPT_NAME_BEST)

        if schedule: schedule.step()
        if train.has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model


def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer, device=None):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = train.AverageMeter()
    top1 = train.AverageMeter()
    top5 = train.AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
            if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = train.has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
        else torch.nn.CrossEntropyLoss()

    has_custom_adv_loss = train.has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = {
            'constraint': args.constraint,
            'eps': eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
            'random_start': args.random_start,
            'custom_loss': adv_criterion,
            'random_restarts': random_restarts,
            'use_best': bool(args.use_best)
        }

    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        # measure data loading time
        if device is not None:
            target = target.to(device)
        else:
            target = target.cuda(non_blocking=True)
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            if train.has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
            else:
                prec1, prec5 = train.helpers.accuracy(model_logits, target, topk=(1, maxk))
                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except Exception as e:
            logger.debug(f"L427: {e}")
            train.warnings.warn('Failed to calculate the accuracy.')

        reg_term = 0.0
        if train.has_attr(args, "regularizer"):
            reg_term = args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with train.amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()
            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = train.make_grid(inp[:15, ...])
            adv_grid = train.make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR
        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Reg term: {reg} ||'.format(epoch, prec, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))

        # USER-DEFINED HOOK
        if train.has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)
    del target, inp
    return top1.avg, losses.avg