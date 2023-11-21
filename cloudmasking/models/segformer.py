# from mmseg.datasets import build_dataset
import torch.nn as nn
from torch.nn.functional import sigmoid
import torchvision.transforms.functional as F
from mmseg.models import build_segmentor
from mmengine.config import Config
from mmengine.logging import MMLogger

logger = MMLogger.get_instance(name='MMLogger',logger_name='Logger')
logger.setLevel('ERROR')

class SegFormer(nn.Module):
    def __init__(self):
        super(SegFormer, self).__init__()
        cfg = Config.fromfile('cloudmasking/models/configs/segformer/segformer_mit-b1_512x512_160k_ade20k.py')
        cfg.model.decode_head.num_classes = 2
        self.model = build_segmentor(cfg.model)
        self.model.init_weights()

    def forward(self, x):
        z = self.model(x)
        z = sigmoid(z)
        z = F.resize(z, x.shape[2:]) # Using bilinear interpolation as default
        return z

def get_model():

    # Load configuration
    # cfg = Config.fromfile('cloudmasking/models/configs/segformer/segformer_mit-b1_512x512_160k_ade20k.py')
    # cfg.model.decode_head.num_classes = 2
    #
    # model = build_segmentor(cfg.model)
    # model.init_weights()
    # return model
    return SegFormer()
