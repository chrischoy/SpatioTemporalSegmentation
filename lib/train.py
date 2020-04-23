import logging
import os.path as osp

import torch
from torch import nn

from tensorboardX import SummaryWriter

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
    Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

from MinkowskiEngine import SparseTensor


def validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn):
  v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config, transform_data_fn)
  writer.add_scalar('validation/mIoU', v_mIoU, curr_iter)
  writer.add_scalar('validation/loss', v_loss, curr_iter)
  writer.add_scalar('validation/precision_at_1', v_score, curr_iter)

  return v_mIoU


def train(model, data_loader, val_data_loader, config, transform_data_fn=None):
  device = get_torch_device(config.is_cuda)
  # Set up the train flag for batch normalization
  model.train()

  # Configuration
  writer = SummaryWriter(log_dir=config.log_dir)
  data_timer, iter_timer = Timer(), Timer()
  data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
  losses, scores = AverageMeter(), AverageMeter()

  optimizer = initialize_optimizer(model.parameters(), config)
  scheduler = initialize_scheduler(optimizer, config)
  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

  writer = SummaryWriter(log_dir=config.log_dir)

  # Train the network
  logging.info('===> Start training')
  best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

  if config.resume:
    checkpoint_fn = config.resume + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      curr_iter = state['iteration'] + 1
      epoch = state['epoch']
      model.load_state_dict(state['state_dict'])
      if config.resume_optimizer:
        scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
        optimizer.load_state_dict(state['optimizer'])
      if 'best_val' in state:
        best_val_miou = state['best_val']
        best_val_iter = state['best_val_iter']
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

  data_iter = data_loader.__iter__()
  while is_training:
    for iteration in range(len(data_loader) // config.iter_size):
      optimizer.zero_grad()
      data_time, batch_loss = 0, 0
      iter_timer.tic()

      for sub_iter in range(config.iter_size):
        # Get training data
        data_timer.tic()
        coords, input, target = data_iter.next()

        # For some networks, making the network invariant to even, odd coords is important
        coords[:, :3] += (torch.rand(3) * 100).type_as(coords)

        # Preprocess input
        color = input[:, :3].int()
        if config.normalize_color:
          input[:, 1:] = input[:, 1:] / 255. - 0.5
        sinput = SparseTensor(input, coords).to(device)

        data_time += data_timer.toc(False)

        # Feed forward
        inputs = (sinput,) if config.wrapper_type == 'None' else (sinput, coords, color)
        # model.initialize_coords(*init_args)
        soutput = model(*inputs)
        # The output of the network is not sorted
        target = target.long().to(device)

        loss = criterion(soutput.F, target.long())

        # Compute and accumulate gradient
        loss /= config.iter_size
        batch_loss += loss.item()
        loss.backward()

      # Update number of steps
      optimizer.step()
      scheduler.step()

      data_time_avg.update(data_time)
      iter_time_avg.update(iter_timer.toc(False))

      pred = get_prediction(data_loader.dataset, soutput.F, target)
      score = precision_at_one(pred, target)
      losses.update(batch_loss, target.size(0))
      scores.update(score, target.size(0))

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
        debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
            epoch, curr_iter,
            len(data_loader) // config.iter_size, losses.avg, lrs)
        debug_str += "Score {:.3f}\tData time: {:.4f}, Total iter time: {:.4f}".format(
            scores.avg, data_time_avg.avg, iter_time_avg.avg)
        logging.info(debug_str)
        # Reset timers
        data_time_avg.reset()
        iter_time_avg.reset()
        # Write logs
        writer.add_scalar('training/loss', losses.avg, curr_iter)
        writer.add_scalar('training/precision_at_1', scores.avg, curr_iter)
        writer.add_scalar('training/learning_rate', scheduler.get_lr()[0], curr_iter)
        losses.reset()
        scores.reset()

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

      # Validation
      if curr_iter % config.val_freq == 0:
        val_miou = validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn)
        if val_miou > best_val_miou:
          best_val_miou = val_miou
          best_val_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                     "best_val")
        logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

        # Recover back
        model.train()

      if curr_iter % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

      # End of iteration
      curr_iter += 1

    epoch += 1

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()

  # Save the final model
  checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
  val_miou = validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn)
  if val_miou > best_val_miou:
    best_val_miou = val_miou
    best_val_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
  logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))
