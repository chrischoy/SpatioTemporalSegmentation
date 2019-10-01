import torch.nn as nn

import MinkowskiEngine as ME

from models.model import Model, NetworkType
from models.modules.common import ConvType, NormType, get_norm, conv, sum_pool
from models.modules.resnet_block import BasicBlock, Bottleneck


class ResNetBase(Model):
  BLOCK = None
  LAYERS = ()
  INIT_DIM = 64
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = 32
  HAS_LAST_BLOCK = False
  CONV_TYPE = ConvType.HYPERCUBE

  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    assert self.BLOCK is not None
    assert self.OUT_PIXEL_DIST > 0

    super(ResNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)

    self.network_initialization(in_channels, out_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, config, D):

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    dilations = config.dilations
    bn_momentum = config.bn_momentum
    self.inplanes = self.INIT_DIM
    self.conv1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        D=D)

    self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[0], 1))
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[1], 1))
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[2], 1))
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[3], 1))

    if self.NETWORK_TYPE == NetworkType.CLASSIFICATION:
      self.glob_avg = ME.MinkowskiGlobalPooling(dimension=D)
      if self.HAS_LAST_BLOCK:
        self.final1 = nn.Linear(self.inplanes, self.inplanes, bias=False)
        self.bnfinal1 = nn.BatchNorm1d(self.inplanes)

        self.final2 = nn.Linear(self.inplanes, self.inplanes, bias=False)
        self.bnfinal2 = nn.BatchNorm1d(self.inplanes)

      self.final = nn.Linear(self.inplanes, out_channels, bias=True)
    else:
      self.final = conv(
          self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              D=self.D))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    if self.NETWORK_TYPE == NetworkType.CLASSIFICATION:
      if self.HAS_LAST_BLOCK:
        res = self.glob_avg(x)
        x = self.final1(res)
        x = self.bnfinal1(x)
        x = self.relu(x)

        x = self.final2(x)
        x = self.bnfinal2(x)
        x += res
        x = self.relu(x)
      else:
        x = self.glob_avg(x)

    x = self.final(x)
    return x


class ResNet14(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 23, 3)


class STResNetBase(ResNetBase):

  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(STResNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STResNet14(STResNetBase, ResNet14):
  pass


class STResNet18(STResNetBase, ResNet18):
  pass


class STResNet34(STResNetBase, ResNet34):
  pass


class STResNet50(STResNetBase, ResNet50):
  pass


class STResNet101(STResNetBase, ResNet101):
  pass


class STResTesseractNetBase(STResNetBase):
  CONV_TYPE = ConvType.HYPERCUBE


class STResTesseractNet14(STResTesseractNetBase, STResNet14):
  pass


class STResTesseractNet18(STResTesseractNetBase, STResNet18):
  pass


class STResTesseractNet34(STResTesseractNetBase, STResNet34):
  pass


class STResTesseractNet50(STResTesseractNetBase, STResNet50):
  pass


class STResTesseractNet101(STResTesseractNetBase, STResNet101):
  pass


# Classification networks
class ResClassificationNetBase(ResNetBase):
  NETWORK_TYPE = NetworkType.CLASSIFICATION


class ResClassificationNet14(ResNet14, ResClassificationNetBase):
  pass


class ResClassificationNet18(ResNet18, ResClassificationNetBase):
  pass


class ResClassificationNet34(ResNet34, ResClassificationNetBase):
  pass


class ResClassificationNet50(ResNet50, ResClassificationNetBase):
  pass


class ResClassificationNet101(ResNet101, ResClassificationNetBase):
  pass


class STResClassificationNet14(STResNetBase, ResClassificationNet14):
  pass


class STResClassificationNet18(STResNetBase, ResClassificationNet18):
  pass


class STResClassificationNet34(STResNetBase, ResClassificationNet34):
  pass


class STResClassificationNet50(STResNetBase, ResClassificationNet50):
  pass


class STResClassificationNet101(STResNetBase, ResClassificationNet101):
  pass


class STResTesseractClassificationNet14(STResTesseractNetBase, STResClassificationNet14):
  pass


class STResTesseractClassificationNet18(STResTesseractNetBase, STResClassificationNet18):
  pass


class STResTesseractClassificationNet34(STResTesseractNetBase, STResClassificationNet34):
  pass


class STResTesseractClassificationNet50(STResTesseractNetBase, STResClassificationNet50):
  pass


class STResTesseractClassificationNet101(STResTesseractNetBase, STResClassificationNet101):
  pass


class ResClassificationNetLB14(ResClassificationNet14):
  HAS_LAST_BLOCK = True


class ResClassificationNetLB18(ResClassificationNet18):
  HAS_LAST_BLOCK = True


class ResClassificationNetLB34(ResClassificationNet34):
  HAS_LAST_BLOCK = True


class ResClassificationNetLB50(ResClassificationNet50):
  HAS_LAST_BLOCK = True


class ResClassificationNetLB101(ResClassificationNet101):
  HAS_LAST_BLOCK = True


class STResClassificationNetLB14(STResClassificationNet14):
  HAS_LAST_BLOCK = True


class STResClassificationNetLB18(STResClassificationNet18):
  HAS_LAST_BLOCK = True


class STResClassificationNetLB34(STResClassificationNet34):
  HAS_LAST_BLOCK = True


class STResClassificationNetLB50(STResClassificationNet50):
  HAS_LAST_BLOCK = True


class STResClassificationNetLB101(STResClassificationNet101):
  HAS_LAST_BLOCK = True


class STResTesseractClassificationNetLB14(STResTesseractClassificationNet14):
  HAS_LAST_BLOCK = True


class STResTesseractClassificationNetLB18(STResTesseractClassificationNet18):
  HAS_LAST_BLOCK = True


class STResTesseractClassificationNetLB34(STResTesseractClassificationNet34):
  HAS_LAST_BLOCK = True


class STResTesseractClassificationNetLB50(STResTesseractClassificationNet50):
  HAS_LAST_BLOCK = True


class STResTesseractClassificationNetLB101(STResTesseractClassificationNet101):
  HAS_LAST_BLOCK = True
