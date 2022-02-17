import torch
import torch.nn as nn

class SEREncoder(nn.Module):		#Speech Emotion Recognition Model
   def __init__(self, hparams, device="cpu"):
      super().__init__()
      self.hparams = hparams
      self.device = device
      self.encoder = nn.ModuleList()
      self.encoder.extend(
                                [nn.Conv2d(1, 128, [7,7], 1, padding='same'),   #input needs to be of shape [batch_size, num_channels, n_mels, n_frames]
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(413696, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048,1024)]
                                         )
      self.output = nn.Linear(1024,6)       #the output layer receives input from encoder
      
      #self.neuralnetwork.append(nn.Conv2d(1, 128, [3,3], 1))	#know the shape of this output
      #self.neuralnetwork.append(nn.ReLU())
      
      #self.neuralnetwork.append(nn.Flatten())
      #output shape: [128 * hparams.n_mels * n_frames=101] = 413696
      
      #self.neuralnetwork.append(nn.Linear(413696, 6))		# 6 = total emotions we have to label
      #self.neuralnetwork.append(nn.Softmax())      

   def forward(self, inputs):   	#
      # input shape: [batch_size, n_mels, n_frames]
      #x = torch.unsqueeze(inputs, dim=1)	#unsqueeze adds 1 to dim 1. New Shape = [batch_size, num_channels=1, n_mels, n_frames]
      x = inputs
      for module in self.encoder:
         x = module(x)      
      return x 	#x -> logits

class SERContrastiveModel(nn.Module):		
   def __init__(self, hparams, device="cpu"):
      super().__init__()
      self.hparams = hparams
      self.device = device
      self.projection = nn.ModuleList()     #projection layer
      self.projection.extend(
                                [nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512,128),
                                 nn.ReLU()]
                                         )

   def forward(self, inputs):   	# will get input from SERModel's encoder
      # input shape: [batch_size, n_mels, n_frames]
      #x = torch.unsqueeze(inputs, dim=1)	#unsqueeze adds 1 to dim 1. New Shape = [batch_size, num_channels=1, n_mels, n_frames]
      x=inputs
      for module in self.projection:
         x = module(x)      
      return x 	#x -> logits
