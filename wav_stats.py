import argparse
import numpy as np

from hparams import Hyperparameters
from audio_utils import load_audio_file
from file_utils import recursive_filter_wav_files


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--data_dir", type=str, default="data", help="A data directory")
   parser.add_argument("-ss", "--samp_size", type=int, default=1, help="The sample size in seconds")
   parser.add_argument("-sr", "--samp_rate", type=int, default=16000, help="The sampling rate in samples per second (Hz)")
   args = parser.parse_args()
   return args

#filter data_list to return all wav file paths that are at least sample_size or longer
def filtered_wav_files(data_list, sr, sample_size):
   filtered_data = []
   for file in data_list:
      wav, _ = load_audio_file(file, sr)
      if len(wav) >= sample_size:
         #print(f"adding wav of length {len(wav)}")
         filtered_data.append(file)         
   return filtered_data

def create_time_bins(times, max_time):	#fix this to include right
   max_time = int(max_time)
   bins = np.zeros(max_time+1)
   for time in times:
      bins[int(time)] += 1
   return bins


#returns the time length of each .wav file  
def get_time_length(data_list, sr):
   print("In get_time_length...")
   times =[]
   for file in data_list:
      y, sr = load_audio_file(file, sr)
      #print(f"len(y)/sr = {len(y)/sr}")
      #print(f"len(y) = {len(y)}")
      times.append(len(y)/sr)
   return times
   
#count how many of each emotion there is in our data -> see if it is evenly distributed 
def get_label_bins(data_list): 
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
   print(f"args = {args}")
   data_list = recursive_filter_wav_files(args.data_dir)
   sampling_rate = args.samp_rate
   sample_size = args.samp_size * sampling_rate
   print(f"len(data_list) = {len(data_list)}")
   print(f"sample_size = {sample_size}")
   print(f"sampling_rate - {sampling_rate}")
   
   filtered_data_list = filtered_wav_files(data_list, sampling_rate, sample_size)
   print(f"len(filtered_data_list) = {len(filtered_data_list)}")
   
   times = get_time_length(filtered_data_list, sampling_rate)
   
   min_data = np.min(times)
   max_data = np.max(times)
   mean_data = np.mean(times)
   
   print("min data length (s) =", min_data)
   print("max data length (s) =", max_data)
   print("avg data length (s) =", mean_data)
   
   time_bins = create_time_bins(times, max_data)
   
   #view time bins
   for i, x in enumerate(time_bins):
      print(i, x)
   print("sum of all bins=", np.sum(time_bins))
   
   #view emotion bins
   label_bins = get_label_bins(filtered_data_list)
   for pair in label_bins.items():
      print(pair)   
  
  
if __name__ == "__main__":
   main()