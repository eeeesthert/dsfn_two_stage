from torch import nn
from .repconv import RepConv
class ShallowFeatureExtractor(nn.Sequential):
 def __init__(self,c=24):super().__init__(RepConv(3,c),RepConv(c,c),RepConv(c,c))
