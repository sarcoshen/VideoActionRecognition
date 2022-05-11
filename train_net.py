#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import model_builder
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter

from tensorboardX import SummaryWriter
log_writer = SummaryWriter('train_log/')

logger = logging.get_logger(__name__)

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    Loss = 0.0
    Top1_err, Top3_err = 0.0, 0.0

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        preds = model(inputs)

          
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        # Compute the loss.
        loss = loss_fun(preds, labels)
        

        # 20 classes
        # 0-qianshou,1-kissing,2-scuffling,3-driving,4-eating,5-hugging
        # 6-chiqiang,7-huishou,8-dancingballet,9-tangodancing,10-chidao
        # 11-singing,12-playingbasketball,13-smoking,14-dadianhua,15-dancingqunti
        # 16-playingguitar,17-ridinghorse,18-swimming,19-shavinghead
        """
        labels_weights = np.array([1.34,1.19,0.58,0.50,0.84,0.79,0.90,1.44,0.94,0.96,1.68,1.51,1.93,1.39,1.0,1.3,1.01,1.0,1.39,1.19])
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(weight=torch.from_numpy(labels_weights).float(),reduction="mean")        
        loss_fun.cuda()
        loss = loss_fun(preds, labels)
        """
        """             
        labels_weights = np.array([1.0, 1.0, 1.0, 0.5, 0.8, 1.0,
                                   0.8, 1.0, 1.0, 1.0, 1.2,
                                   1.0, 1.0, 1.0, 1.0, 1.0,
                                   1.0, 1.0, 1.0, 1.0])
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(weight=torch.from_numpy(labels_weights).float(),reduction="mean")
        loss_fun.cuda()
        loss = loss_fun(preds, labels)        
        """
        #criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(labels_weights).float())
        #criterion.cuda()
        #loss = criterion(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 3))
        top1_err, top3_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top3_err = du.all_reduce(
                [loss, top1_err, top3_err]
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top3_err = (
            loss.item(),
            top1_err.item(),
            top3_err.item(),
        )

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            top1_err, top3_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
        )

        Loss += loss
        Top1_err += top1_err
        Top3_err += top3_err

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    log_writer.add_scalars('Training', {"training_loss": float(Loss) / data_size}, cur_epoch)
    # log_writer.add_scalars('Training', {"training loss": float(Loss) / data_size,
    #                                     "topt_error": float(Top1_err) / data_size,
    #                                     "top5_error": float(Top5_err) / data_size}, cur_epoch)


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    Loss = 0.0
    data_size = len(val_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
        else:
            preds = model(inputs)

            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top3_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top3_err = du.all_reduce([top1_err, top3_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top3_err = top1_err.item(), top3_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err, top3_err, inputs[0].size(0) * cfg.NUM_GPUS
            )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    log_writer.add_scalars('testing', {"testing loss": float(Loss) / data_size}, cur_epoch)


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

 
         
    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
        print("yyyyyyyyyyyyyyyyyyyyyyyy")
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    else:
        start_epoch = 0
        


    #start_epoch = 0


    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
      
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        # # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)





