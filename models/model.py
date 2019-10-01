from enum import Enum

import torch
from MinkowskiEngine import MinkowskiNetwork

from lib.utils import HashTimeBatch


class NetworkType(Enum):
  """
  Classification or segmentation.
  """
  SEGMENTATION = 0, 'SEGMENTATION',
  CLASSIFICATION = 1, 'CLASSIFICATION'

  def __new__(cls, value, name):
    member = object.__new__(cls)
    member._value_ = value
    member.fullname = name
    return member

  def __int__(self):
    return self.value


class Model(MinkowskiNetwork):
  """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """
  OUT_PIXEL_DIST = -1
  NETWORK_TYPE = NetworkType.SEGMENTATION

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    super(Model, self).__init__(D)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.config = config

  def permute_label(self, label, max_label):
    if not isinstance(self.OUT_PIXEL_DIST, (list, tuple)):
      assert self.OUT_PIXEL_DIST > 0, "OUT_PIXEL_DIST not set"
    return super(Model, self).permute_label(label, max_label, self.OUT_PIXEL_DIST)


class SpatialModel(Model):
  """
  Base network for all spatial sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    assert D == 3, "Num dimension not 3"
    super(SpatialModel, self).__init__(in_channels, out_channels, config, D, **kwargs)

  def initialize_coords(self, coords):
    # In case it has temporal axis
    if coords.size(1) > 4:
      spatial_coord, time, batch = coords[:, :3], coords[:, 3], coords[:, 4]
      time_batch = HashTimeBatch()(time, batch)
      coords = torch.cat((spatial_coord, time_batch.unsqueeze(1)), dim=1)

    super(SpatialModel, self).initialize_coords(coords)


class SpatioTemporalModel(Model):
  """
  Base network for all spatio temporal sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    assert D == 4, "Num dimension not 4"
    super(SpatioTemporalModel, self).__init__(in_channels, out_channels, config, D, **kwargs)


class HighDimensionalModel(Model):
  """
  Base network for all spatio (temporal) chromatic sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    assert D > 4, "Num dimension smaller than 5"
    super(HighDimensionalModel, self).__init__(in_channels, out_channels, config, D, **kwargs)
