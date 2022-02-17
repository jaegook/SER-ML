from label_builder import LabelBuilder

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

def create_label_builder(data_list):
   lb = LabelBuilder()
   for file in data_list:
      file_info = file.split('_')
      emotion_label = file_info[2]
      lb.add_label(emotion_label)
   return lb

