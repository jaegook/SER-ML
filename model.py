import torch
import torch.nn as nn

class SERSupConModel(nn.Module):		#Speech emotion recognition, supervised contrastive model
   def __init__(self, hparams, device="cpu"):
      super().__init__()
      self.hparams = hparams
      self.device = device
      self.encoder = nn.ModuleList()
      self.encoder.extend(
                                [nn.Conv2d(1, 128, [7,7], 1, padding='same'),   #input needs to be of shape [batch_size, num_channels, n_mels, n_frames]
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1642496, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048,1024)]
                                         )
      self.projection = nn.ModuleList()     #projection layer
      self.projection.extend(
                                [nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512,128),
                                 nn.ReLU()]
                                         )
      
      self.classifier = nn.Linear(1024,6)       #the output layer receives input from encoder
      
      #self.neuralnetwork.append(nn.Conv2d(1, 128, [3,3], 1))	#know the shape of this output
      #self.neuralnetwork.append(nn.ReLU())
      
      #self.neuralnetwork.append(nn.Flatten())
      #output shape: [128 * hparams.n_mels * n_frames=101] = 413696
      
      #self.neuralnetwork.append(nn.Linear(413696, 6))		# 6 = total emotions we have to label
      #self.neuralnetwork.append(nn.Softmax())      

   def forward(self, inputs, contrastive):   	#
      # input shape: [batch_size, n_mels, n_frames]
      #x = torch.unsqueeze(inputs, dim=1)	#unsqueeze adds 1 to dim 1. New Shape = [batch_size, num_channels=1, n_mels, n_frames]
      x = inputs
      for encoder_module in self.encoder:
         print("in encoder")
         x = encoder_module(x) 
      if contrastive:
         print("in contrastive")
         x = nn.functional.normalize(x, p=2, dim=1)
         for proj_module in self.projection:
            print("in projection")
            x = proj_module(x)
      else:
         print("in mode = sup_con, classifier")
         x = x.detach()
         x = self.classifier(x)  
         #print(f"type(x)={type(x)}")
      return x 	#x -> logits

class SERClassifierModel(nn.Module):		#Speech Emotion Recognition Model
   def __init__(self, hparams, device="cpu"):
      super().__init__()
      self.hparams = hparams
      self.device = device
      self.encoder = nn.ModuleList()
      self.encoder.extend(
                                [nn.Conv2d(1, 128, [7,7], 1, padding='same'),   #input needs to be of shape [batch_size, num_channels, n_mels, n_frames]
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1642496, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048,1024)]
                                         )
      
      self.classifier = nn.Linear(1024,6)       #the output layer receives input from encoder
      
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
      for encoder_module in self.encoder:
         print("in encoder")
         x = encoder_module(x)
      x = self.classifier(x)   
      print("after self.classifier(x)")         
      return x 	#x -> logits
