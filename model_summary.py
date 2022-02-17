#import torch
#from pytorch_model_summary import summary
from torchsummary import summary
from model import SERModel
from hparams import Hyperparameters

hparams = Hyperparameters()
model = SERModel(hparams)
summary(model, (1,32,101))       #summary(model, (channels, H, w))