import logging
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import spatial

from lib.utils import read_txt, fast_hist, per_class_iu
from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
import lib.transforms as t
from lib.datasets.preprocessing.stanford_3d import Stanford3DDatasetConverter


class StanfordVoxelizationDatasetBase:
  CLIP_SIZE = None
  CLIP_BOUND = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  NUM_LABELS = 14
  IGNORE_LABELS = (10,)  # remove stairs, following SegCloud

  # CLASSES = [
  #     'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
  #     'table', 'wall', 'window'
  # ]

  IS_FULL_POINTCLOUD_EVAL = True

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.TrainVal: 'trainval.txt',
      DatasetPhase.Test: 'test.txt'
  }

  def test_pointcloud(self, pred_dir):
    print('Running full pointcloud evaluation.')
    # Join room by their area and room id.
    room_dict = defaultdict(list)
    for i, data_path in enumerate(self.data_paths):
      area, room = data_path.split(os.sep)
      room, _ = os.path.splitext(room)
      room_id = '_'.join(room.split('_')[:-1])
      room_dict[(area, room_id)].append(i)
    # Test independently for each room.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    pred_list = sorted(os.listdir(pred_dir))
    hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
    for room_idx, room_list in enumerate(room_dict.values()):
      print(f'Evaluating room {room_idx} / {len(room_dict)}.')
      # Join all predictions and query pointclouds of split data.
      pred = np.zeros((0, 4))
      pointcloud = np.zeros((0, 7))
      for i in room_list:
        pred = np.vstack((pred, np.load(os.path.join(pred_dir, pred_list[i]))))
        pointcloud = np.vstack((pointcloud, self.load_ply(i)[0]))
      # Deduplicate all query pointclouds of split data.
      pointcloud = np.array(list(set(tuple(l) for l in pointcloud.tolist())))
      # Run test for each room.
      pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
      _, result = pred_tree.query(pointcloud[:, :3])
      ptc_pred = pred[result, 3].astype(int)
      ptc_gt = pointcloud[:, -1].astype(int)
      if self.IGNORE_LABELS:
        ptc_pred = self.label2masked[ptc_pred]
        ptc_gt = self.label2masked[ptc_gt]
      hist += fast_hist(ptc_pred, ptc_gt, self.NUM_LABELS)
      # Print results.
      ious = []
      print('Per class IoU:')
      for i, iou in enumerate(per_class_iu(hist) * 100):
        unmasked_idx = self.label2masked.tolist().index(i)
        result_str = f'\t{Stanford3DDatasetConverter.CLASSES[unmasked_idx]}:\t'
        if hist.sum(1)[i]:
          result_str += f'{iou}'
          ious.append(iou)
        else:
          result_str += 'N/A'  # Do not print if data not in ground truth.
        print(result_str)
      print(f'Average IoU: {np.nanmean(ious)}')


class StanfordVoxelizationDataset(StanfordVoxelizationDatasetBase, VoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  CLIP_BOUND = None
  VOXEL_SIZE = 5  # 5cm

  # Augmentation arguments
  ELASTIC_DISTORT_PARAMS = ((20, 100), (80, 320))
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               augment_data=True,
               elastic_distortion=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_root = config.stanford3d_online_path
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    data_paths = [d.split()[0] for d in data_paths]
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))

    VoxelizationDataset.__init__(
        self,
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)


class StanfordVoxelization2cmDataset(StanfordVoxelizationDataset):

  VOXEL_SIZE = 2


def test(config):
  """Test point cloud data loader.
  """
  from torch.utils.data import DataLoader
  from lib.utils import Timer
  timer = Timer()
  DatasetClass = StanfordVoxelization2cmDataset
  transformations = [
      t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
      t.ChromaticAutoContrast(),
      t.ChromaticTranslation(config.data_aug_color_trans_ratio),
      t.ChromaticJitter(config.data_aug_color_jitter_std),
      t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
  ]

  dataset = DatasetClass(
      config,
      input_transform=t.Compose(transformations),
      augment_data=True,
      cache=True,
      elastic_distortion=True)

  data_loader = DataLoader(
      dataset=dataset,
      collate_fn=t.cfl_collate_fn_factory(limit_numpoints=False),
      batch_size=4,
      shuffle=True)

  # Start from index 1
  iter = data_loader.__iter__()
  for i in range(100):
    timer.tic()
    data = iter.next()
    print(timer.toc())


if __name__ == '__main__':
  from config import get_config
  config = get_config()

  test(config)
