import torch
from torch.utils.data.sampler import Sampler


class InfSampler(Sampler):
  """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

  def __init__(self, data_source, shuffle=False):
    self.data_source = data_source
    self.shuffle = shuffle
    self.reset_permutation()

  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)

  next = __next__  # Python 2 compatibility
