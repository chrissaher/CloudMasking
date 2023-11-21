# from mmseg.datasets import build_dataset
import torch.nn as nn
from torch.nn.functional import sigmoid
import torchvision.transforms.functional as F
from mmseg.models import build_segmentor
from mmengine.config import Config
from mmengine.logging import MMLogger

logger = MMLogger.get_instance(name='MMLogger',logger_name='Logger')
logger.setLevel('ERROR')

class Deeplabv3(nn.Module):
    def __init__(self):
        super(Deeplabv3, self).__init__()
        cfg = Config.fromfile('cloudmasking/models/configs/deeplabv3/deeplabv3_r50-d8_512x512_4x4_160k_coco-stuff164k.py')
        cfg.model.decode_head.num_classes = 2
        cfg.model.auxiliary_head.num_classes = 2
        self.model = build_segmentor(cfg.model)
        self.model.init_weights()

    def forward(self, x):
        z = self.model(x)
        z = sigmoid(z)
        z = F.resize(z, x.shape[2:]) # Using bilinear interpolation as default
        return z

def get_model():
    return Deeplabv3()
