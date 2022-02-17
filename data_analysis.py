def analyze_crema_data(data_list):
   # find number of unique actors
   # find number of times each emotion appears
   actor_set = set()
   emotion_dict = {}
   for data in data_list:
      data_split = data.split("_")
      actor_set.add(data_split[0])
      if data_split[2] in emotion_dict:
         emotion_dict[data_split[2]] += 1
      else:
         emotion_dict[data_split[2]] = 1
   for pair in emotion_dict.items():
      print(pair)
   print("Number of unique actors:", len(actor_set))