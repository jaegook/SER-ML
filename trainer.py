import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import evaluator
import os.path

from audio_utils import load_audio_file, plot_time_domain, test
from dataset import create_dataloader, create_contrastive_dataloader
from model import SEREncoder, SERContrastiveModel
from hparams import Hyperparameters
from losses import ContrastiveLoss

class SERTrainer:

   def __init__(self, hparams, device="cpu"):
      self.hparams = hparams
      self.device = device
      self.encoder = SEREncoder(hparams, device)
      self.contrastive_model = SERContrastiveModel(hparams, device)
      self.loss_fn = nn.CrossEntropyLoss
      self.contrastive_loss_fn = ContrastiveLoss()
      self.optimizer = optim.Adam(self.encoder.parameters(), hparams.init_lr, betas=[0.9,0.999])	# nn.module calls parameters()
      self.lr_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, hparams.init_lr, hparams.end_lr)

   def train_step(self, x, label):
      logits = self.model(x)	# nn.module passes x to model.forward()
      loss = self.loss_fn(logits, label)
      loss.backward()
      self.optimizer.step()
      self.lr_scheduler.step()
      self.optimizer.zero_grad()
      return loss, logits
   def contrastive_train_step(self, x, labels):
      encoder_output = self.encoder(x)                                      #outputs 1024 feature vectors for each example, in our case 120 [120,1024]
      #print("type(ENCODER_OUTPUT)={}, ENCODER_OUTPUT.size()={}".format(type(encoder_output),encoder_output.size()))
      normalized_encoder_output = nn.functional.normalize(encoder_output, p=2, dim=1)
      #print("type(normalized_encoder_output)={}, normalized_encoder_output.size()={}".format(type(normalized_encoder_output), normalized_encoder_output.size()))

      projection_output = self.contrastive_model(normalized_encoder_output)            #projection_output shape -> [120,128]
      #print("projection_output.shape=", projection_output.shape)      
      
      
      elem_per_group = len(projection_output)//self.hparams.num_contrastive_samples #we want 2 groups when we split dim=0
      projection_output = torch.split(projection_output, elem_per_group, dim=0) #shape: [120,128] -> ([60,128],[60,128])
      #for step, x in enumerate(projection_output):
      #   print("{}: x.size()={} ".format(step, x.size()))
      
      projection_output = torch.stack(projection_output, dim=1) #shape: ([60,128],[60,128]) -> [60,2,128]
      #print("After Stack: projection_output.size()=",projection_output.size())
      
      projection_output = torch.split(projection_output, self.hparams.contrastive_batch_size, dim=0) #shape: [60,2,128] -> ([10,2,128], ..., [10,2,128]) there's 6 of these in the tuple
      #for step,x in enumerate(projection_output):
      #   print("{}: x.size()={} ".format(step, x.size()))
      
      projection_output = torch.cat(projection_output, dim=2) #([10,2,128], ..., [10,2,128]) => 6 of these -> [10,2,768]
      #print("projection_output.size()=", projection_output.size())
      
      projection_output = torch.split(projection_output, 1, dim=0) #[10,2,768] -> ([1,2,768], ..., [1,2,768]) -> 10 in the tuple
      #for step,x in enumerate(projection_output):
      # print("{}: x.size()={} ".format(step, x.size()))
      
      projection_output_list = []
      for step,x in enumerate(projection_output):
         #print("{}: x.size()={} ".format(step, x.size()))
         y = torch.split(x,128,dim=2) # we want to split [1,2,768] -> ([1,2,128], ..., [1,2,128]) -> 6 [1,2,128]'s in the tuple
         #print("after torch.split(x,128,dim=2) y.size()=", y.size())
         #for i in y:
         #   print("i.size()=",i.size())
         y = torch.cat(y, dim=0)
         #print("y.size()=",y.size()) # we want to concat 6 [1,2,128]'s across dim=0 to get -> [6,2,128]
         projection_output_list.append(y)
     
      label_list = torch.unbind(labels, dim=0)
      
      train_loss = self.contrastive_loss_fn(projection_output_list, label_list, self.hparams.temperature)
      
      train_loss.backward()
      self.optimizer.step()
      #scheduluer step
      self.optimizer.zero_grad()
      
      
      return train_loss
      
   def contrastive_train_loop(self, train_dataloader, valid_dataloader, label_builder):
      self.encoder.to(self.device)
      self.contrastive_model.to(self.device)
      best_loss = float("inf")
      for epoch in range(self.hparams.num_contrastive_epochs):
         total_loss = 0.0
         self.encoder.train()
         self.contrastive_model.train()
         start_time = time.time()
         for step, (pos_negs, labels) in enumerate(train_dataloader):
            
            
            #query aka anchor is in pos_negs
            pos_negs = pos_negs.to(self.device)     # pos_negs ->[10,2,32,606] 
            
                          
            #manipulate the shape so we can feed it encoder who accepts shape -> [batch, 1, 32, 101]             
            y = torch.split(pos_negs, self.hparams.num_frames, dim=3)           #split pos_negs:[10,2,32,606] -> we have 6 [10,2,32,101]            
            y = torch.cat(y, dim=0)                                             #concat pos_negs on dim=0 ->shape [60, 2, 32, 101]                
            y = torch.split(y,1,dim=1)                                          #split the examples on dim=1 -> shape [60,2,32,101] -> [60,1,32,101], [60,1,32,101]
            encoder_input = torch.cat(y, dim=0)                                #concat the two split data across dim=0 -> shape [120,1,32,101]
            
            labels = labels.to(self.device)
            
            
            train_loss = self.contrastive_train_step(encoder_input, labels)
            total_loss += train_loss
            if step + 1 % self.hparams.contrastive_display_step == 0:
               print("Epoch = {}, Step = {}, Training Loss = {}".format(epoch, step, train_loss))
               end_time = time.time()
               print(f"Elapsed time for {step}: {(end_time-start_time)/60}")
            if step + 1 % self.hparams.contrastive_validation_step == 0:
               avg_loss = total_loss / (step + 1)
               contrastive_model = self.make_contrastive_model_path(epoch, avg_loss, self.hparams.model_dir)
               save_contrastive_model(self.encoder, self.contrastive_model, contrastive_model, epoch, avg_loss)
         end_time = time.time()
         print("time for 1 epoch (min) =",(end_time-start_time)/60)
         print("Validating...")
         avg_loss = total_loss / (step + 1)
         print("Epoch ={} average loss = {}".format(epoch, avg_loss))
         #evaluator.evaluate_and_display(true_label_list, pred_list, label_builder)
         cont_model_path = self.make_contrastive_model_path(epoch, avg_loss, self.hparams.model_dir)
         self.save_contrastive_model(self.encoder, self.contrastive_model, cont_model_path, epoch, avg_loss)
         """
         start_time = time.time()
         val_loss, acc, precision, recall, fscore, support = evaluator.evaluate(valid_dataloader, self.model, self.loss_fn, self.device)
         end_time = time.time()
         print("time for validation (min) =", (end_time - start_time)/60)
         print("Epoch = {}, Validation Loss = {}, Validation Accuracy = {}".format(epoch, val_loss, acc))
         #print precision and recall for each class
         for i, (p, r, f, s) in enumerate(zip(precision.tolist(),recall.tolist(), fscore.tolist(), support.tolist())):
            print("for class {}, precision = {}, recall = {}, fscore = {}, support = {}".format(label_builder.get_label(i),p,r,f,s))
         if val_loss < best_loss:
            best_loss = val_loss
            print("best loss = {}".format(best_loss))            
            model_path = self.make_model_path(epoch, best_loss, self.hparams.model_dir)				#create this fn
            #self.save_model(self.model, model_path, epoch, best_loss)		                                #create this model
        """
   def train_loop(self, train_dataloader, valid_dataloader, label_builder):
      self.model.to(self.device)
      best_loss = float("inf")
      for epoch in range(self.hparams.num_epochs):
         total_loss = 0.0
         pred_list = []
         true_label_list = []
         self.model.train()
         start_time = time.time()
         for step, (x, label) in enumerate(train_dataloader):
            x = x.to(self.device)
            label = label.to(self.device)
            train_loss, logits = self.train_step(x, label)
            preds = torch.argmax(logits, dim=1)
            true_labels = torch.argmax(label, dim=1)
            pred_list += preds.tolist()
            true_label_list += true_labels.tolist()
            total_loss += train_loss
            if step % self.hparams.display_step == 0:
               print("Epoch = {}, Step = {}, Training Loss = {}".format(epoch, step, train_loss))
         end_time = time.time()
         print("time for 1 epoch (min) =",(end_time-start_time)/60)
         print("Validating...")
         avg_loss = total_loss / (step + 1)
         print("Epoch ={} average loss = {}".format(epoch, avg_loss))
         #evaluator.evaluate_and_display(true_label_list, pred_list, label_builder)
         """
         start_time = time.time()
         val_loss, acc, precision, recall, fscore, support = evaluator.evaluate(valid_dataloader, self.model, self.loss_fn, self.device)
         end_time = time.time()
         print("time for validation (min) =", (end_time - start_time)/60)
         print("Epoch = {}, Validation Loss = {}, Validation Accuracy = {}".format(epoch, val_loss, acc))
         #print precision and recall for each class
         for i, (p, r, f, s) in enumerate(zip(precision.tolist(),recall.tolist(), fscore.tolist(), support.tolist())):
            print("for class {}, precision = {}, recall = {}, fscore = {}, support = {}".format(label_builder.get_label(i),p,r,f,s))
         if val_loss < best_loss:
            best_loss = val_loss
            print("best loss = {}".format(best_loss))            
            model_path = self.make_model_path(epoch, best_loss, self.hparams.model_dir)				#create this fn
            #self.save_model(self.model, model_path, epoch, best_loss)		                                #create this model
        """           
   
   def make_model_path(self, epoch, loss, model_dir):
      model_path = f"{model_dir}/SER_SUPCON_{epoch}_{loss}.pth"
      return model_path
      
   def save_model(self, model, model_path, epoch, best_loss):   			#need to save optimizer, best_loss, current_epoch, lr_scheduler, model.state_dict()
      state = {
               'epoch': epoch,
               'state_dict': model.state_dict(),
               'optimizer': self.optimizer.state_dict(),
               'scheduler': self.lr_scheduler.state_dict(),
               'best_loss': best_loss,
               }
      print("saving to {}...".format(model_path))
      torch.save(state, model_path)
       
   def make_contrastive_model_path(self, epoch, loss, model_dir):
      model_path = f"{model_dir}/SER_SUPCON_{epoch}_{loss}.pth"
      return model_path
      
   def save_contrastive_model(self, encoder, contrastive_model, model_path, epoch, best_loss):
      state = {
               'epoch': epoch,
               'encoder_state_dict': encoder.state_dict(),
               'contrastive_state_dict': contrastive_model.state_dict(),
               'optimizer': self.optimizer.state_dict(),
               'scheduler': self.lr_scheduler.state_dict(),
               'best_loss': best_loss,
               }
      print("saving to {}...".format(model_path))
      torch.save(state, model_path)
   
def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
   parser.add_argument("-d", "--data_dirs", nargs="+", required=True, help="A data directory")
   args = parser.parse_args()
   return args

def train(train_dataloader, valid_dataloader, hparams, label_builder):
   device = "cuda" if torch.cuda.is_available() else "cpu"      #use gpu if available
   print(f"using {device}")
   trainer = SERTrainer(hparams, device)
   trainer.contrastive_train_loop(train_dataloader, valid_dataloader, label_builder)
   #trainer.train_loop(train_dataloader, valid_dataloader, label_builder)

def check_environment(args, hparams):
   if not os.path.isdir(hparams.model_dir):
      print("creating {} directory...".format(hparams.model_dir))
      os.mkdir(hparams.model_dir)
   if not os.path.isdir(hparams.labels_dir):
      print("creating {} directory...".format(hparams.labels_dir))
      os.mkdir(hparams.labels_dir)
   if not os.path.isdir(args.data_dirs[0]):
      print("{} does not exist".format(args.data_dirs[0]))
   if not os.path.isdir(args.data_dirs[1]):
      print("{} does not exist".format(args.data_dirs[1]))
      
def main():
   args = parse_args()
   hparams = Hyperparameters()
   check_environment(args, hparams)
   
   train_contrastive_dl, lb = create_contrastive_dataloader(args.data_dirs[0], hparams)
   valid_contrastive_dl, lb = create_contrastive_dataloader(args.data_dirs[1], hparams)
   #train_dataloader, lb = create_dataloader(args.data_dirs[0], hparams)  
   #valid_dataloader, lb = create_dataloader(args.data_dirs[1], hparams)
   
   train(train_contrastive_dl, valid_contrastive_dl, hparams, lb)
   #train(train_dataloader, valid_dataloader, hparams, lb)

  
if __name__ == "__main__":
   main()