from .synthia import SynthiaCVPR15cmVoxelizationDataset, SynthiaCVPR30cmVoxelizationDataset, \
    SynthiaAllSequencesVoxelizationDataset
from .stanford import StanfordVoxelizationDataset, StanfordVoxelization2cmDataset
from .scannet import ScannetVoxelizationDataset, ScannetVoxelization2cmDataset

DATASETS = [
    StanfordVoxelizationDataset, StanfordVoxelization2cmDataset, ScannetVoxelizationDataset,
    ScannetVoxelization2cmDataset, SynthiaCVPR15cmVoxelizationDataset,
    SynthiaCVPR30cmVoxelizationDataset, SynthiaAllSequencesVoxelizationDataset
]


def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass
