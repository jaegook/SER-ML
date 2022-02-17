import argparse
import numpy as np

from hparams import Hyperparameters
from audio_utils import load_audio_file
from dataset import VoiceConversionDataset

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
   parser.add_argument("-d", "--data_dir", type=str, default="data", help="A data directory")
   args = parser.parse_args()
   return args

def create_time_bins(times, max_time):	#fix this to include right
   bins = np.zeros(int(max_time+1))
   for time in times:
      bins[int(time)] += 1
   return bins

def get_time_length(data_list, sr):
   #returns the time length of each .wav file  
   times =[]
   i = 1
   for file in data_list:
      y, sr = load_audio_file(file, sr)
      print("file =", file)
      print("y, sr =", i, y, sr)
      times.append(len(y)/sr)
      i += 1
   return times

def get_emotion_bins(data_list): #count how many of each emotion there is in our data -> see if it is evenly distributed 
   dict = {}
   for file in data_list:
      emotion = file.split("_")[2]
      if emotion in dict:
         dict[emotion] += 1
      else:
         dict[emotion] = 1
   return dict
def main():
   
   args = parse_args()
   hparams = Hyperparameters()
   
   ds = VoiceConversionDataset(args.data_dir, hparams)
   data_list = ds.data_list
   print("data_list = ", data_list)

   times = get_time_length(data_list, hparams.sampling_rate)
   
   min = np.min(times)
   max = np.max(times)
   mean = np.mean(times)
   
   print("min length (s) =", min)
   print("max length (s) =", max)
   print("avg length (s) =", mean)
   
   time_bins = create_time_bins(times, max)
   
   #view time bins
   for i, x in enumerate(time_bins):
      print(i, x)
   print("sum of all bins=", np.sum(time_bins))
   
   #view emotion bins
   emotion_bins = get_emotion_bins(data_list)
   for pair in emotion_bins.items():
      print(pair)   
  
  
if __name__ == "__main__":
   main()