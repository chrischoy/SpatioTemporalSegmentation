# Change dataloader multiprocess start method to anything not fork
import open3d as o3d

import torch.multiprocessing as mp
try:
  mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
  pass

import os
import sys
import json
import logging
from easydict import EasyDict as edict

# Torch packages
import torch

# Train deps
from config import get_config

from lib.test import test
from lib.train import train
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

from models import load_model, load_wrapper

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def main():
  config = get_config()
  if config.resume:
    json_config = json.load(open(config.resume + '/config.json', 'r'))
    json_config['resume'] = config.resume
    config = edict(json_config)

  if config.is_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found")
  device = get_torch_device(config.is_cuda)

  logging.info('===> Configurations')
  dconfig = vars(config)
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  DatasetClass = load_dataset(config.dataset)
  if config.test_original_pointcloud:
    if not DatasetClass.IS_FULL_POINTCLOUD_EVAL:
      raise ValueError('This dataset does not support full pointcloud evaluation.')

  if config.evaluate_original_pointcloud:
    if not config.return_transformation:
      raise ValueError('Pointcloud evaluation requires config.return_transformation=true.')

  if (config.return_transformation ^ config.evaluate_original_pointcloud):
    raise ValueError('Rotation evaluation requires config.evaluate_original_pointcloud=true and '
                     'config.return_transformation=true.')

  logging.info('===> Initializing dataloader')
  if config.is_train:
    train_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.train_phase,
        num_workers=config.num_workers,
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.batch_size,
        limit_numpoints=config.train_limit_numpoints)

    val_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_val_workers,
        phase=config.val_phase,
        augment_data=False,
        shuffle=True,
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False)
    if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color

    num_labels = train_data_loader.dataset.NUM_LABELS
  else:
    test_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_workers,
        phase=config.test_phase,
        augment_data=False,
        shuffle=False,
        repeat=False,
        batch_size=config.test_batch_size,
        limit_numpoints=False)
    if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color

    num_labels = test_data_loader.dataset.NUM_LABELS

  logging.info('===> Building model')
  NetClass = load_model(config.model)
  if config.wrapper_type == 'None':
    model = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                      count_parameters(model)))
  else:
    wrapper = load_wrapper(config.wrapper_type)
    model = wrapper(NetClass, num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(
        wrapper.__name__ + NetClass.__name__, count_parameters(model)))

  logging.info(model)
  model = model.to(device)

  if config.weights == 'modelzoo':  # Load modelzoo weights if possible.
    logging.info('===> Loading modelzoo weights')
    model.preload_modelzoo()

  # Load weights if specified by the parameter.
  elif config.weights.lower() != 'none':
    logging.info('===> Loading weights: ' + config.weights)
    state = torch.load(config.weights)
    if config.weights_for_inner_model:
      model.model.load_state_dict(state['state_dict'])
    else:
      if config.lenient_weight_loading:
        matched_weights = load_state_with_same_shape(model, state['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(matched_weights)
        model.load_state_dict(model_dict)
      else:
        model.load_state_dict(state['state_dict'])

  if config.is_train:
    train(model, train_data_loader, val_data_loader, config)
  else:
    test(model, test_data_loader, config)


if __name__ == '__main__':
  __spec__ = None
  main()
