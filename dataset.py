import torch
import random
import numpy as np
import os

from label_builder import LabelBuilder
from torch.utils.data import Dataset, DataLoader
from file_utils import recursive_filter_wav_files
from audio_utils import load_audio_file, wav_to_mel
from crema_utils import create_label_builder

class SER(Dataset):
   def __init__(self, data_dir, hparams):
      self.data_list = recursive_filter_wav_files(data_dir)	# data_list contains filenames of our wavefile
      print("data_list size before filter =", len(self.data_list))
      self.hparams = hparams
      self.filter_data_list()
      print("data_list size after filter =", len(self.data_list))

      #create_label_dict -> should change its name and return a list of labels and have LabelBuilder build dictionary
      #self.crema_label_dict = create_label_dict(self.data_list) 

      label_dict_path = self.create_label_dict_path()
      if os.path.exists(label_dict_path):
         self.label_builder = LabelBuilder()
         print("loading label_to_id dictionary...")
         self.label_builder.load(label_dict_path)     
      else:
         self.label_builder = create_label_builder(self.data_list)
         print("saving label_to_id dictionary...")
         self.label_builder.save(label_dict_path)

      
      #change __getitem__ to use LabelBuilder
   def __len__(self):
      return len(self.data_list)

   def __getitem__(self, index):
      wav_file_path = self.data_list[index]
      
      # extract emotion dict key
      emotion_key = wav_file_path.split('_')[2]
      one_hot_index = self.label_builder.get_label_id(emotion_key)
      
      #print("emotion_key=", emotion_key)
      #print("one_hot_index=", one_hot_index)      

      # create one hot vector
      #label = np.zeros(self.label_builder.count())
      #label[one_hot_index] = 1
      label = one_hot_index
      
      wav, sr = load_audio_file(wav_file_path, self.hparams.sampling_rate)
     
      
      # we have to sample only part of wav data, hparams.sample_size
      # find the offset so we get full sample_size from wav
      offset = len(wav) - self.hparams.sample_size

      # get random starting point from 0 to offset to sample
      start = random.randint(0, offset)

      sampled_wav = wav[start:start+self.hparams.sample_size]
      assert len(sampled_wav) == self.hparams.sample_size        # length of the sampled_wav must equal hparams.sample_size or else give error message
      
      # get and return the mel spectrogram
      log_mel_spectrogram = wav_to_mel(sampled_wav, self.hparams.sampling_rate, self.hparams.frame_size, self.hparams.hop_length, self.hparams.n_mels)
      # ->log_mel_spectrogram shape = [n_mels, n_frames]
                  
      return log_mel_spectrogram, label 

   def filter_data_list(self):
      filtered_data = []
      for file in self.data_list:
         wav, sr = load_audio_file(file, self.hparams.sampling_rate)
         if len(wav) >= self.hparams.sample_size:
            #print("len(y)/sr, sample_size/sampling_rate =", len(y)/sr, self.hparams.sample_size/self.hparams.sampling_rate)
            filtered_data.append(file)
      self.data_list = filtered_data

   def create_label_dict_path(self):
      label_dict_path = f"{self.hparams.labels_dir}/label_to_id.json"
      return label_dict_path


class SERContrastive(SER):
   def __init__(self, data_dir, hparams):
      super().__init__(data_dir, hparams)
      self.data_dict = {}
      self.build_data_dict()
   
   def __getitem__(self, index):
      wav_file_path = self.data_list[index]
      wav, sr = load_audio_file(wav_file_path, self.hparams.sampling_rate)
      
      #find offset so we can get full sample of size = hparams.sample_size
      offset = len(wav) - self.hparams.sample_size
      start = random.randint(0, offset)
      sampled_wav = wav[start:start+self.hparams.sample_size]
      assert(len(sampled_wav) == self.hparams.sample_size)
      
      #get log_mel_spectrogram to feed to network
      log_mel_spectrogram = wav_to_mel(sampled_wav, self.hparams.sampling_rate, self.hparams.frame_size, self.hparams.hop_length, self.hparams.n_mels)
      
      log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
      assert(log_mel_spectrogram.shape == (1,32,101))
      
      #extract label and get one hot index
      emotion_label = wav_file_path.split('_')[2]
      one_hot_index = self.label_builder.get_label_id(emotion_label)
      
      
      #create one hot vector
      #label = np.zeros(self.label_builder.count())
      #label[one_hot_index] = 1
      
      labels = [one_hot_index]
      
      #pos_neg_examples will hold anchor, and (self.hparams.num_contrastive_samples - 1) pos examples, and (self.hparams.num_contrastive_samples) neg examples per emotion
      pos_neg_examples = []
      anchor = log_mel_spectrogram
      
      #key will be 1 of the 6 emotions, vals[0] = current starting index, vals[1] = list of data for the key
      for key, vals in self.data_dict.items():
         assert(len(vals[1]) > self.hparams.num_contrastive_samples)
         #start_index = vals[0]
         start = vals[0]
         if key == emotion_label:       # we need one less positive example
            #end_index = start_index + (self.hparams.num_contrastive_samples - 2)
            end = start + (self.hparams.num_contrastive_samples - 1)
         else:
            #end_index = start_index + self.hparams.num_contrastive_samples - 1
            end = start + self.hparams.num_contrastive_samples
            labels.append(self.label_builder.get_label_id(key))
         
         file_list = vals[1]
         
         #if we are starting from the beginning of list shuffle data
         if start == 0:
            random.shuffle(self.data_dict[key][1])
            file_list = self.data_dict[key][1]
         #if there is not enough data in the list to get num_contrastive_samples, shuffle the data and start from the beginning
         if end > len(file_list):
            random.shuffle(self.data_dict[key][1])
            file_list = self.data_dict[key][1]
            self.data_dict[key][0] = 0
            start = self.data_dict[key][0]    
            if key == emotion_label:
               end = start + (self.hparams.num_contrastive_samples - 1)
            else:
               end = start + self.hparams.num_contrastive_samples
        
            
         list = []
         if key == emotion_label:
            list.append(anchor)
         for i in range(start, end):
            #if key == emotion_label:
            #   print("KEY==EMOTION_LABEL, START={}, END={}".format(start, end))
            if key == emotion_label and file_list[i] == wav_file_path:  #we don't want duplicate data so pick a random data 
               #print("KEY == EMOTION AND file_list[i] == wav_file_path, i = {}",format(i))
               if len(file_list) - end > 0:
                  #print("IF: len(file_list)-1 = {}, end_index={}".format(len(file_list)-1, end_index))  
                  #print("IF: index range: {} - {}".format(end_index,len(file_list) - 1))
                  assert(end <= len(file_list) - 1)
                  idx = random.randint(end, len(file_list) - 1)
                  #print("in if i={}, idx={}".format(i,idx))
               else:
                  #print("ELSE: index range: 0 - {}".format(start_index))
                  assert(start > 0)
                  idx = random.randint(0, start-1)
                  #print("in else i={}, idx={}".format(i,idx))
               wav, sr = load_audio_file(file_list[idx], self.hparams.sampling_rate)
            else:
               #print("index = {} len(file_list)={}".format(i, len(file_list)))
               wav, sr = load_audio_file(file_list[i], self.hparams.sampling_rate)
            offset = len(wav) - self.hparams.sample_size
            beg = random.randint(0, offset)
            sampled_wav = wav[beg:beg+self.hparams.sample_size]
            assert(len(sampled_wav) == self.hparams.sample_size)
            lm_spectrogram = wav_to_mel(sampled_wav, self.hparams.sampling_rate, self.hparams.frame_size, self.hparams.hop_length, self.hparams.n_mels)
            lm_spectrogram = np.expand_dims(lm_spectrogram, axis=0) #add a dimension for n_channels
            assert(lm_spectrogram.shape == (1,32,101))
            #print("type(lm_spectrogram)={} lm_spectrogram.shape={}".format(type(lm_spectrogram), lm_spectrogram.shape))
            
            list.append(lm_spectrogram)
            self.data_dict[key][0] += 1
         
         list = np.vstack(list) #after np.vstack list shape is [self.hparams.num_contrastive_samples,32,101]
         #print("type(list)={}, len(list)={}, list.shape={}".format(type(list), len(list), list.shape))
         if key == emotion_label:            
            if len(pos_neg_examples)==0:
               pos_neg_examples.append(list)
            else:
               pos_neg_examples.append(pos_neg_examples[0])
               pos_neg_examples[0] = list
         else:
            pos_neg_examples.append(list)
      #print("before np.concat: pos_neg_examples.shape=", np.shape(pos_neg_examples))
      #for pos_negs in pos_neg_examples:
      #   print("pos_negs.shape=", pos_negs.shape)
      pos_neg_examples = np.concatenate(pos_neg_examples, axis=2)
      labels = np.array(labels)
      #print("IN __getitem__: pos_neg_examples.shape={}, labels.shape={}".format(pos_neg_examples.shape, labels.shape))
     
      return pos_neg_examples, labels      #pos_neg_examples->shape[2,32,606] [n_contrastive_samples, n_mels, n_frames], label->shape[6,] 
               
         
      
   def build_data_dict(self):
      for file in self.data_list:
         emotion = file.split('_')[2]
         if emotion not in self.data_dict:
            #self.data_dict[0] = starting index, self.data_dict[1] = list of files
            self.data_dict[emotion] = [0,[file]]
         else:
            self.data_dict[emotion][1].append(file)


def create_dataloader(data_dir, hparams):
   ds = SER(data_dir, hparams)
   #print("batch_size =", hparams.batch_size)
   #print("ds size =", len(ds.data_list))
   
   data_loader = DataLoader(ds, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers)
 
   return data_loader, ds.label_builder		
   
def create_contrastive_dataloader(data_dir, hparams):
   ds = SERContrastive(data_dir, hparams)
   data_loader = DataLoader(ds, batch_size=hparams.contrastive_batch_size, shuffle=True, drop_last=True)
   return data_loader, ds.label_builder


def test():
   hparams = Hyperparameters()
   dl, lb = create_contrastive_dataloader("./test", hparams)
   for step, (pos_neg_examples, labels) in enumerate(dl):
      print("step=",step)     
      print("type(pos_neg_examples)={}, pos_neg_examples.size()={}".format(type(pos_neg_examples), pos_neg_examples.size()))
      print("type(labels)={}, len(labels)={}".format(type(labels), len(labels)))
      for i, label in enumerate(labels):
         print("{}: {}".format(i,label))
      #for i, x in enumerate(x[0]):
      #   print("{}) type(x) ={} len(x)={}".format(i, type(x), len(x)))
      #for i in x[0]:
      #   print(type(i), i.size())


if __name__ == "__main__":
   from hparams import Hyperparameters
   test()
   
