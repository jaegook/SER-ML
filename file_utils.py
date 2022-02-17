import os

#step 0: data_list = []
#step 1: given a root directory: "data/"
#step 2: list the root directory. It returns list of directories and files in the root directory
#step 3: Iterate through the list of directories and files
#step 4: check if it's a directory
#step 5: if yes, create a new path by merging it with the root directory
#step 6: recrusively call recursive(new_root_dir) returns data_list of wav files in new_root_dir
#step 7: else check if its a wav file
#step 8: if yes, merge it with the root dir and add it to an data_list
#step 9: return data_list with all .wav files in root "data/"

def recursive_filter_wav_files(data_dir):
   data_list = []
   file_list = os.listdir(data_dir)

   for data_file in file_list:
      root_ext = os.path.splitext(data_file)

      if os.path.isdir(os.path.join(data_dir,data_file)):
         new_root_dir = os.path.join(data_dir, data_file)
         data_list = recursive_filter_wav_files(new_root_dir) + data_list

      elif root_ext[1] == '.wav':
         wav_filepath = os.path.join(data_dir, data_file)
         data_list.append(wav_filepath)

   return data_list