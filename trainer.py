import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import evaluator
import os.path

from audio_utils import load_audio_file, plot_time_domain, test
from dataset import create_dataloader, create_contrastive_dataloader
from model import SERSupConModel, SERClassifierModel
from hparams import Hyperparameters
from losses import ContrastiveLoss

class SERTrainer:

   def __init__(self, hparams, device="cpu"):
      self.hparams = hparams
      self.device = device
      self.ser_classifier = SERClassifierModel(self.hparams, device)
      self.classifier_loss_fn = nn.CrossEntropyLoss()
      self.classifier_optimizer = optim.Adam(self.ser_classifier.parameters(), hparams.init_lr, betas=[0.9,0.999])	# nn.module calls parameters()
      self.lr_scheduler = optim.lr_scheduler.LinearLR(self.classifier_optimizer, hparams.init_lr, hparams.end_lr)


   def train_step(self, x, label):
      logits = self.ser_classifier(x)	# nn.module passes x to model.forward()
      print(f"logits.shape ={label.shape}")
      print(f"label.shape={label.shape}")
      print("computing loss")
      loss = self.classifier_loss_fn(logits, label)
      print("backwards prop")
      loss.backward()
      print(f"loss = {loss}")
      print("optimizing")
      self.classifier_optimizer.step()
      print("scheduler step")
      self.lr_scheduler.step()
      print("zero grad")
      self.classifier_optimizer.zero_grad()
      return loss, logits
      
   def train_loop(self, train_dataloader, valid_dataloader, label_builder):
      self.ser_classifier.to(self.device)
      best_loss = float("inf")
      print("Starting classifier training...")
      for epoch in range(self.hparams.num_epochs):
         total_loss = 0.0
         pred_list = []
         true_label_list = []
         self.ser_classifier.train()    #can i put this outside epoch for loop?
         epoch_start_time = time.time()
         print(f"starting training for epoch:{epoch}")
         for step, (x, label) in enumerate(train_dataloader):
            x = x.to(self.device)
            label = label.to(self.device)
            x = torch.unsqueeze(x,1)
            #print(f"x.size() = {x.size()}")
            train_loss, logits = self.train_step(x, label)
            #print(f"type(logits) = {type(logits)}, logits.shape = {logits.shape}")
            preds = torch.argmax(logits, dim=1)
            true_labels = torch.argmax(label, dim=1)
            pred_list += preds.tolist()
            true_label_list += true_labels.tolist()
            total_loss += train_loss
            if step % self.hparams.display_step == 0:
               print("Epoch = {}, Step = {}, Training Loss = {}".format(epoch, step, train_loss))
            if (step+1) % self.hparams.validation_step == 0:
               best_loss = self.validate_and_save_model(valid_dataloader, epoch, best_loss, label_builder, self.ser_classifier, self.classifier_optimizer)
         
         epoch_end_time = time.time()
         print("time for 1 epoch (min) =",(epoch_end_time-epoch_start_time)/60)
         avg_loss = total_loss / (step + 1)
         print("Epoch ={} average loss = {}".format(epoch, avg_loss))
         print("Validating...")
         best_loss = self.validate_and_save_model(valid_dataloader, epoch, best_loss, label_builder, self.ser_classifier, self.classifier_optimizer)
  
   def validate_and_save_model(self, valid_dataloader, epoch, best_loss, label_builder, model, optimizer):
         print(f"Validating for epoch: {epoch}")
         val_start_time = time.time()
         val_loss, metrics = evaluator.evaluate(valid_dataloader, model, self.loss_fn, self.device)
         val_end_time = time.time()
         print("time for validation (min) =", (val_end_time - val_start_time)/60)
         print("Epoch = {}, Validation Loss = {}".format(epoch, val_loss))
         evaluator.print_metrics(metrics, label_builder)
         
         if val_loss < best_loss:
            best_loss = val_loss
            print("best loss = {}".format(best_loss))            
            model_path = self.make_model_path(epoch, best_loss, self.hparams.model_dir)				            #
            self.save_model(model, model_path, epoch, best_loss, optimizer)		                                
         
         return best_loss
         
   def make_model_path(self, epoch, loss, model_dir):
      model_path = f"{model_dir}/SER_{epoch}_{loss}.pth"
      return model_path
      
   def save_model(self, model, model_path, epoch, best_loss, optimizer):   			#need to save optimizer, best_loss, current_epoch, lr_scheduler, model.state_dict()
      state = {
               'epoch': epoch,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': self.lr_scheduler.state_dict(),
               'best_loss': best_loss,
               }
      print("saving to {}...".format(model_path))
      torch.save(state, model_path)
       

class SERSupConTrainer(SERTrainer):
   def __init__(self, hparams, device="cpu"):
      super.__init__(hparams, device)
      self.ser_supcon = SERSupConModel(hparams, device)
      self.contrastive_loss_fn = ContrastiveLoss()
      self.contrastive_optimizer = optim.Adam(self.ser_supcon.parameters(), self.hparams.init_lr, betas=[0.9,0.999])
   
   def train_step(self, x, labels):
      logits = self.ser_supcon(x, contrastive=False)
      loss = self.classifier_loss_fn(logits, label)
      loss.backward()
      self.classifier_optimizer.step()  #question: how is this optimizer connected to the ser_supcon? we gave it ser_classifier's parameters
      self.lr_scheduler.step()
      self.classifier_optimizer.zero_grad()
      return loss, logits
      pass
   
   def train_loop(train_dataloader, valid_dataloader, label_builder):
      self.ser_supcon.to(self.device)
      best_loss = float("inf")
      print("starting contrastive classifier training...")
      for epoch in range(self.hparams.num_epochs):
         total_loss = 0.0
         pred_list = []
         true_label_list = []
         self.ser_supcon.train()
         epoch_start_time = time.time()
         for step, (x, label) in enumerate(train_dataloader):
            x.to(self.device)
            label.to(self.device)
            x = torch.unsqueeze(x, 1)
            train_loss, logits = self.train_step(x, label)
            preds = torch.argmax(logits, dim=1)
            true_labels = torch.argmax(label, dim=1)
            pred_list += preds.tolist()
            true_label_list += true_labels.tolist()
            total_loss += train_loss
            if step % self.hparams.display_step == 0:
               print(f"Epoch: {epoch}, Step: {step}, Train loss: {train_loss}")
            if (step + 1) % self.hparams.validation_step == 0:
              validate_and_save_model(valid_dataloader, epoch, best_loss, label_builder, self.ser_supcon, self.classifier_optimizer)
         epoch_end_time = time.time()
         print("time for 1 epoch (min) =",(epoch_end_time-epoch_start_time)/60)
         avg_loss = total_loss / (step + 1)
         print("Epoch ={} average loss = {}".format(epoch, avg_loss))
         print("Validating...")
         best_loss = self.validate_and_save_model(valid_dataloader, epoch, best_loss, label_builder, self.ser_classifier, self.classifier_optimizer)
   
   def contrastive_train_step(self, x, labels):
      output = self.ser_supcon(x, contrastive=True)   #outputs 120 feature vectors for each example, in our case [120,128]
      elem_per_group = len(output)//self.hparams.num_contrastive_samples #we want 2 groups when we split dim=0
      projection_output = torch.split(output, elem_per_group, dim=0) #shape: [120,128] -> ([60,128],[60,128])
      projection_output = torch.stack(projection_output, dim=1) #shape: ([60,128],[60,128]) -> [60,2,128]
      projection_output = torch.split(projection_output, self.hparams.contrastive_batch_size, dim=0) #shape: [60,2,128] -> ([10,2,128], ..., [10,2,128]) there's 6 of these in the tuple
      projection_output = torch.cat(projection_output, dim=2) #([10,2,128], ..., [10,2,128]) => 6 of these -> [10,2,768]    
      projection_output = torch.split(projection_output, 1, dim=0) #[10,2,768] -> ([1,2,768], ..., [1,2,768]) -> 10 in the tuple
      projection_output_list = []
      for step,x in enumerate(projection_output):
         y = torch.split(x,128,dim=2)   # we want to split [1,2,768] -> ([1,2,128], ..., [1,2,128]) -> 6 [1,2,128]'s in the tuple
         y = torch.cat(y, dim=0)        # we want to concat 6 [1,2,128]'s across dim=0 to get -> [6,2,128]
         projection_output_list.append(y)
      label_list = torch.unbind(labels, dim=0)
      train_loss = self.contrastive_loss_fn(projection_output_list, label_list, self.hparams.temperature, self.device)
      train_loss.backward()
      self.contrastive_optimizer.step()
      self.contrastive_optimizer.zero_grad()
      return train_loss

   def contrastive_train_loop(self, train_dataloader, valid_dataloader, label_builder):
      self.ser_supcon.to(self.device)
      best_loss = float("inf")
      print("Starting contrastive training...")
      for epoch in range(self.hparams.num_contrastive_epochs):
         total_loss = 0.0
         self.ser_supcon.train()
         start_time = time.time()
         #print(f"starting training for epoch:{epoch}")
         for step, (pos_negs, labels) in enumerate(train_dataloader):
            #query aka anchor is in pos_negs
            pos_negs = pos_negs.to(self.device)     # pos_negs ->[batch_size, num_contrastive_samples, n_mels, n_frames*(num_neg_examples + 1 positive examples)]             
            #manipulate the shape so we can feed it encoder who accepts shape -> [batch, 1, n_mels, n_frames]             
            y = torch.split(pos_negs, self.hparams.num_frames, dim=3)           #split pos_negs:[10,2,32,606] -> we have 6 [10,2,32,101]            
            y = torch.cat(y, dim=0)                                             #concat pos_negs on dim=0 ->shape [60, 2, 32, 101]                
            y = torch.split(y,1,dim=1)                                          #split the examples on dim=1 -> shape [60,2,32,101] -> [60,1,32,101], [60,1,32,101]
            encoder_input = torch.cat(y, dim=0)                                #concat the two split data across dim=0 -> shape [120,1,32,101]            
            labels = labels.to(self.device)
            train_loss = self.contrastive_train_step(encoder_input, labels)
            total_loss += train_loss
            if step % self.hparams.contrastive_display_step == 0:
               print("Epoch = {}, Step = {}, Training Loss = {}".format(epoch, step, train_loss))
               end_time = time.time()
               print(f"Elapsed time for {step}: {(end_time-start_time)/60}")
            if (step + 1) % self.hparams.contrastive_validation_step == 0:
               avg_loss = total_loss / (step + 1)
               contrastive_model = self.make_contrastive_model_path(epoch, avg_loss, self.hparams.model_dir)
               self.save_model(self.ser_supcon, contrastive_model, epoch, avg_loss, self.contrastive_optimizer)
         end_time = time.time()
         print("time for 1 epoch (min) =",(end_time-start_time)/60)
         print("Validating...")
         avg_loss = total_loss / (step + 1)
         print("Epoch ={} average loss = {}".format(epoch, avg_loss))
         #evaluator.evaluate_and_display(true_label_list, pred_list, label_builder)
         cont_model_path = self.make_contrastive_model_path(epoch, avg_loss, self.hparams.model_dir)
         self.save_model(self.ser_supcon, cont_model_path, epoch, avg_loss, self.contrastive_optimizer)
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
   def make_contrastive_model_path(self, epoch, loss, model_dir):
      model_path = f"{model_dir}/SER_SUPCON_{epoch}_{loss}.pth"
      return model_path
      
def train(args, hparams):
   device = "cuda" if torch.cuda.is_available() else "cpu"      #use gpu if available
   print(f"using {device}")
   if hparams.mode == "classifier":
      trainer = SERTrainer(hparams, device)
   elif hparams.mode == "sup_con":
      trainer = SERSupConTrainer(hparams, device)
   if hparams.mode == "sup_con":
      train_contrastive_dataloader, label_builder = create_contrastive_dataloader(args.data_dirs[0], hparams)
      valid_contrastive_dataloader, _ = create_contrastive_dataloader(args.data_dirs[1], hparams)
      train_dataloader, _ = create_dataloader(args.data_dirs[0], hparams)  
      valid_dataloader, _ = create_dataloader(args.data_dirs[1], hparams)
      
      trainer.contrastive_train_loop(train_contrastive_dataloader, valid_contrastive_dataloader, label_builder)
      trainer.train_loop(train_dataloader, valid_dataloader, label_builder)
   elif hparams.mode == "classifier":
      train_dataloader, label_builder = create_dataloader(args.data_dirs[0], hparams)  
      valid_dataloader, _ = create_dataloader(args.data_dirs[1], hparams)
      trainer.train_loop(train_dataloader, valid_dataloader, label_builder)

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

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
   parser.add_argument("-d", "--data_dirs", nargs="+", required=True, help="A data directory")
   parser.add_argument("-m", "--mode", type=str, default="sup_con", help="training mode either: classifier or sup_con") 
   args = parser.parse_args()
   return args
   
def main():
   args = parse_args()
   hparams = Hyperparameters()
   hparams.mode = args.mode
   check_environment(args, hparams)
   train(args, hparams)
   
  
if __name__ == "__main__":
   main()