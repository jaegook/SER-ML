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
      label = np.zeros(self.label_builder.count())
      label[one_hot_index] = 1
      
      
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
      #print(f"label= {label}")
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
      anchor_wav_file_path = self.data_list[index]
      anchor_wav, sr = load_audio_file(anchor_wav_file_path, self.hparams.sampling_rate)
      
      #find offset so we can get full sample of size = hparams.sample_size
      anchor_offset = len(anchor_wav) - self.hparams.sample_size
      start = random.randint(0, anchor_offset)
      sampled_anchor_wav = anchor_wav[start:start+self.hparams.sample_size]
      assert(len(sampled_anchor_wav) == self.hparams.sample_size)
      
      #get log_mel_spectrogram to feed to network
      anchor_log_mel_spectrogram = wav_to_mel(sampled_anchor_wav, self.hparams.sampling_rate, self.hparams.frame_size, self.hparams.hop_length, self.hparams.n_mels)
      
      anchor_log_mel_spectrogram = np.expand_dims(anchor_log_mel_spectrogram, axis=0)
      assert(anchor_log_mel_spectrogram.shape == (1,32,101))
      
      #extract label and get one hot index
      anchor_label_key = anchor_wav_file_path.split('_')[2]
      anchor_label_id = self.label_builder.get_label_id(anchor_label_key)
      
      
      #create one hot vector
      #label = np.zeros(self.label_builder.count())
      #label[one_hot_index] = 1
      
      #label_id_list = [anchor_label_id]
      
      #pos_neg_examples will hold anchor, and (self.hparams.num_contrastive_samples - 1) pos examples, and (self.hparams.num_contrastive_samples) neg examples per emotion
      
      anchor_mels_list = [anchor_log_mel_spectrogram]
      anchor_pair = self.data_dict[anchor_label_key]
      start_index = anchor_pair[0]
      files = anchor_pair[1]
      if start_index + self.hparams.num_contrastive_samples - 1 >= len(files):
         random.shuffle(files)
         start_index = 0
      
      assert(len(files) >= self.hparams.num_contrastive_samples)
      anchor_mels_list += self.get_contrastive_samples(files, start_index, len(anchor_mels_list), anchor_wav_file_path, anchor_label_key)
      if len(anchor_mels_list) < self.hparams.num_contrastive_samples:
         random.shuffle(files)
         start_index = 0
         anchor_mels_list += self.get_contrastive_samples(files, start_index, len(anchor_mels_list), anchor_wav_file_path, anchor_label_key)
      assert(len(anchor_mels_list) == self.hparams.num_contrastive_samples)  

      #print(f"len(anchor_mels_list) should be {self.hparams.num_contrastive_samples}={len(anchor_mels_list)}")
      #update starting index in data_dict
            
      #there should be hparams.num_contrastive_samples log_mels in the list
      #for step, log_mel in enumerate(anchor_mels_list):
      #   print(f"{step + 1}: log_mel.shape should be {self.hparams.num_contrastive_samples}[1,32,101]={log_mel.shape}")
      
      pos_examples = np.concatenate(anchor_mels_list, axis=0)
      #print(f"pos_examples.shape should be [{self.hparams.num_contrastive_samples},32,101]={pos_examples.shape}")  #pos_examples-> shape:[num_contrastive_samples, n_mels, n_frames]
      #print(f"type(pos_examples)={type(pos_examples)}")
      pos_neg_examples = [pos_examples]
      
      #make negative mels:
      neg_label_keys = list(self.data_dict.keys())
      
      idx_to_del = neg_label_keys.index(anchor_label_key)
      del neg_label_keys[idx_to_del]
      
      #print(f"anchor_label_key={anchor_label_key}, neg_label_keys shouldn't contain anchor_label_key->{neg_label_keys}")
      
      random.shuffle(neg_label_keys)
      neg_label_keys = neg_label_keys[:self.hparams.num_neg_examples]
      pos_neg_label_ids = [anchor_label_id]
      for label_key in neg_label_keys:
         neg_label_id = self.label_builder.get_label_id(label_key)
         pos_neg_label_ids.append(neg_label_id)
         neg_pair = self.data_dict[label_key]
         start_index = neg_pair[0]
         files = neg_pair[1]
         if start_index + self.hparams.num_contrastive_samples >= len(files):
            random.shuffle(files)
            start_index = 0
         neg_examples = []
         assert(len(files) >= self.hparams.num_contrastive_samples)
         neg_examples += self.get_contrastive_samples(files, start_index, len(neg_examples), anchor_wav_file_path, label_key)
         assert(len(neg_examples) == self.hparams.num_contrastive_samples)
     
         neg_examples = np.concatenate(neg_examples, axis=0) #neg_examples-> shape:[num_contrastive_samples, n_mels, n_frames]
         #print(f"type(neg_examples)={type(neg_examples)}")
         pos_neg_examples.append(neg_examples)
      
      assert(len(pos_neg_examples) == self.hparams.num_neg_examples + 1)  
      assert(len(pos_neg_label_ids) == self.hparams.num_neg_examples + 1)      
      #print(f"type(pos_neg_examples)={type(pos_neg_examples)}")
      #after for loop pos_neg_examples-> should have num_neg_examples + positve examples of [num_contrastive_samples, n_mels, n_frames]
      #for i, pos_neg in enumerate(pos_neg_examples):
      #   print(f"{i + 1}: pos_neg.shape should be {self.hparams.num_neg_examples + 1} of [{self.hparams.num_contrastive_samples}, 32, 101]={pos_neg.shape}")
      
      pos_neg_examples = np.concatenate(pos_neg_examples, axis=2)    #pos_neg_examples -> shape:[num_contrastive_samples, n_mels, n_frames*(num_neg_examples + 1)]
      pos_neg_label_ids = np.array(pos_neg_label_ids)
      #print(f"pos_neg_examples.shape should be [{self.hparams.num_contrastive_samples},32,{101*(self.hparams.num_neg_examples + 1)}]={pos_neg_examples.shape}")
      #print(f"pos_neg_label_ids.shape should be[{self.hparams.num_neg_examples +1},]={pos_neg_label_ids.shape}")   #pos_neg_label_ids.shape-> [num_neg_examples + 1]
      #print(f"type(pos_neg_examples)={type(pos_neg_examples)}, type(pos_neg_label_ids)={type(pos_neg_label_ids)}")
      #print(pos_neg_label_ids)
      return pos_neg_examples, pos_neg_label_ids
      
   def get_contrastive_samples(self, files, start_index, curr_num_samples, anchor_wav_file_path, label_key):
      mels_list = []
      i = 0
      for file in files[start_index:]:
         if curr_num_samples == self.hparams.num_contrastive_samples:
            break
         if file != anchor_wav_file_path:
            wav, sr = load_audio_file(file, self.hparams.sampling_rate)
            offset = len(wav) - self.hparams.sample_size
            start = random.randint(0,offset)
            sampled_wav = wav[start:start+self.hparams.sample_size]
            assert(len(sampled_wav) == self.hparams.sample_size)
            log_mel = wav_to_mel(sampled_wav, self.hparams.sampling_rate, self.hparams.frame_size, self.hparams.hop_length, self.hparams.n_mels)
            log_mel = np.expand_dims(log_mel, axis=0)
            assert(log_mel.shape == (1,32,101))
            mels_list.append(log_mel)
            curr_num_samples += 1
         i+=1
      #update starting index in data_dict:
      self.data_dict[label_key][0] += i
      return mels_list
      
   def build_data_dict(self):
      random.shuffle(self.data_list)
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
'''
def pad(batch):
   print(f"type(batch), {type(batch)}, {len(batch)}, {type(batch[0])}, {type(batch[1])}")
   print(f"{type(batch[0][0])}, {type(batch[0][1])}")
   print(f"{type(batch[0][0])}, {len(batch[0][0])}")
   print(f"{batch[0][0].shape}, {batch[0][1].shape}")
   print(f"{batch[1][0].shape}, {batch[1][1].shape}")
'''
   
def create_contrastive_dataloader(data_dir, hparams):
   ds = SERContrastive(data_dir, hparams)
   data_loader = DataLoader(ds, batch_size=hparams.contrastive_batch_size, shuffle=True, drop_last=True)  #, collate_fn=pad)
   return data_loader, ds.label_builder


def test():
   hparams = Hyperparameters()
   dl, lb = create_contrastive_dataloader("../test", hparams)
   print("looping through dataloader...")
   for step, (pos_neg_examples, labels) in enumerate(dl):
      print("step=",step)     
      print("type(pos_neg_examples)={}, pos_neg_examples.size()={}".format(type(pos_neg_examples), pos_neg_examples.size()))
      print("type(labels)={}, labels.size()={}".format(type(labels), labels.size()))
      for i, label in enumerate(labels):
         print("{}: {}".format(i,label))

if __name__ == "__main__":
   from hparams import Hyperparameters
   test()
   
