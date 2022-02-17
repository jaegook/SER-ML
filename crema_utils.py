# given a file name, get actor ID and emotion label

def get_file_info(filename):
   emotion_dict = {'SAD' : 'sadness',
                   'ANG' : 'angry',
                   'DIS' : 'disgust',
                   'FEA' : 'fear',
                   'HAP' : 'happy',
                   'NEU' : 'neutral'}
   file_info = filename.split('_')
  
   actor_id = file_info[0].split('/')[-1]
 
   emotion_label = emotion_dict[file_info[2]]
   return (actor_id, emotion_label)

def create_label_dict(data_list):
   dict = {}
   i = 0
   for file in data_list:
      file_info = file.split('_')
      emotion_label = file_info[2]
      if dict.get(emotion_label) is None:
         dict[emotion_label] = i
         i += 1
   return dict

